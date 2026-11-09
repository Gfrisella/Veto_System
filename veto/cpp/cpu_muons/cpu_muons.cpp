// CPU port of faster_muons_torch.propagate_muons_with_alias_sampling (MuonsAndMatter/cuda_muons/faster_muons_torch:
// cuda_muons_field_map.cu + common.cuh), field-map version, one OpenMP iteration per muon. The same arguments, on CPU
// tensors; the same physics step by step (material lookup, alias sampling of the energy loss / pT kick, rotation, RK4
// with the field at the start and end of the step, the big step in the open region). Only the random numbers differ:
// curand's XORWOW per (seed, muon) is replaced by a counter-seeded xoshiro128+, so the result agrees statistically,
// not muon by muon.
//
//   python3 setup.py build_ext --inplace   ->  import cpu_muons; cpu_muons.propagate_muons_with_alias_sampling(...)

#include <torch/extension.h>
#include <omp.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace {

constexpr float MUON_MASS = 0.1056583755f;
constexpr float C_LIGHT = 299792458.0f;
constexpr float E_CHARGE = 1.602176634e-19f;
constexpr float GEV_OVER_C_TO_SI = 5.3443e-19f;
constexpr float BIG_STEP = 0.2f;

enum Material { IRON, CONCRETE, AIR };

struct Arb8 { float neg[4][2], pos[4][2], z_center, dz; };

struct FieldMeta {
    int binsX, binsY, binsZ, stride_y;
    float startX, endX, startY, endY, startZ, endZ, invX, invY, invZ;
};

struct GridMeta { float z_min, z_max, inv_h; int sz; };

struct Hist {
    const float *prob, *c1, *c2, *w1, *w2;
    const int *alias;
};

struct Env {
    Hist mat[2];
    int H;
    float log_start, inv_log_step;
    const float *field;
    FieldMeta fm;
    const Arb8 *arb8s;
    const int *cells, *items;
    GridMeta gm;
    const float *cav;
    bool sym;
};

// xoshiro128+, seeded per (seed, muon) by splitmix64
struct Rng {
    uint32_t s[4];
    Rng(uint64_t seed, uint64_t idx) {
        uint64_t z = seed * 0x9E3779B97F4A7C15ULL ^ (idx + 0x632BE59BD9B4E019ULL);
        for (int i = 0; i < 2; ++i) {
            z += 0x9E3779B97F4A7C15ULL;
            uint64_t x = z;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
            x ^= x >> 31;
            s[2 * i] = uint32_t(x);
            s[2 * i + 1] = uint32_t(x >> 32);
        }
        if (!(s[0] | s[1] | s[2] | s[3])) s[0] = 1;
    }
    static uint32_t rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }
    uint32_t next() {
        const uint32_t r = s[0] + s[3], t = s[1] << 9;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 11);
        return r;
    }
    float uniform() { return float((next() >> 8) + 1) * (1.0f / 16777216.0f); }   // (0, 1], as curand_uniform
};

inline float norm3(const float v[3]) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

inline void rotate_vector(const float u[3], const float d[3], const float P[3], float out[3]) {
    // rotation taking z onto u (Rodrigues), applied to d, added to P
    float ax[3] = {-u[1], u[0], 0.0f};
    float an = norm3(ax);
    if (an == 0) { for (int i = 0; i < 3; ++i) out[i] = P[i] + d[i]; return; }
    for (int i = 0; i < 3; ++i) ax[i] /= an;
    const float th = std::acos(u[2]), s = std::sin(th), c1 = 1 - std::cos(th);
    const float K[3][3] = {{0, -ax[2], ax[1]}, {ax[2], 0, -ax[0]}, {-ax[1], ax[0], 0}};
    float R[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R[i][j] = (i == j) + s * K[i][j] + c1 * (K[i][0] * K[0][j] + K[i][1] * K[1][j] + K[i][2] * K[2][j]);
    for (int i = 0; i < 3; ++i) out[i] = P[i] + R[i][0] * d[0] + R[i][1] * d[1] + R[i][2] * d[2];
}

inline void field_at(const Env &e, float x, float y, float z, float B[3]) {
    const FieldMeta &m = e.fm;
    const float sx = e.sym ? std::fabs(x) : x, sy = e.sym ? std::fabs(y) : y;
    if (sx < m.startX || sx > m.endX || sy < m.startY || sy > m.endY || z < m.startZ || z > m.endZ) {
        B[0] = B[1] = B[2] = 0; return;
    }
    const int i = int(std::nearbyint((sx - m.startX) * m.invX * (m.binsX - 1)));   // __float2int_rn
    const int j = int(std::nearbyint((sy - m.startY) * m.invY * (m.binsY - 1)));
    const int k = int(std::nearbyint((z - m.startZ) * m.invZ * (m.binsZ - 1)));
    const float *f = e.field + size_t(j * m.stride_y + i * m.binsZ + k) * 3;
    float gx = 1, gz = 1;
    if (e.sym) {
        gx = ((x >= 0 && y >= 0) || (x <= 0 && y <= 0)) ? 1.0f : -1.0f;
        gz = ((x >= 0 && y >= 0) || (x <= 0 && y >= 0)) ? 1.0f : -1.0f;
    }
    B[0] = f[0] * gx; B[1] = f[1]; B[2] = f[2] * gz;
}

inline bool inside_arb8(float x, float y, float z, const Arb8 &b) {
    if (std::fabs(z - b.z_center) > b.dz) return false;
    const float f = (z - (b.z_center - b.dz)) / (2.0f * b.dz);
    float v[4][2], sg[4];
    for (int i = 0; i < 4; ++i)
        for (int c = 0; c < 2; ++c) v[i][c] = (1.0f - f) * b.neg[i][c] + f * b.pos[i][c];
    for (int i = 0; i < 4; ++i) {
        const int j = (i + 1) % 4;
        sg[i] = (v[j][0] - v[i][0]) * (y - v[i][1]) - (v[j][1] - v[i][1]) * (x - v[i][0]);
    }
    return (sg[0] >= 0 && sg[1] >= 0 && sg[2] >= 0 && sg[3] >= 0) || (sg[0] <= 0 && sg[1] <= 0 && sg[2] <= 0 && sg[3] <= 0);
}

inline Material material(const Env &e, float x, float y, float z) {
    const float *c = e.cav;
    if ((z <= c[4] && (x <= c[0] || x >= c[1] || y <= c[2] || y >= c[3])) ||
        (z > c[4] && (x <= c[5] || x >= c[6] || y <= c[7] || y >= c[8])))
        return CONCRETE;
    if (z >= e.gm.z_max || z < e.gm.z_min) return AIR;
    if (e.sym) { x = std::fabs(x); y = std::fabs(y); }
    const int cell = int((z - e.gm.z_min) * e.gm.inv_h);
    for (int i = e.cells[cell]; i < e.cells[cell + 1]; ++i)
        if (inside_arb8(x, y, z, e.arb8s[e.items[i]])) return IRON;
    return AIR;
}

inline void deriv(const float s[6], float q, const float B[3], float energy, float d[6]) {
    const float vx = s[3] / energy * C_LIGHT, vy = s[4] / energy * C_LIGHT, vz = s[5] / energy * C_LIGHT;
    d[0] = vx; d[1] = vy; d[2] = vz;
    d[3] = q * (vy * B[2] - vz * B[1]) / GEV_OVER_C_TO_SI;
    d[4] = q * (vz * B[0] - vx * B[2]) / GEV_OVER_C_TO_SI;
    d[5] = q * (vx * B[1] - vy * B[0]) / GEV_OVER_C_TO_SI;
}

inline void rk4_cached(const Env &e, float pos[3], float mom[3], float charge, float h) {
    float s[6] = {pos[0], pos[1], pos[2], mom[0], mom[1], mom[2]};
    const float p = norm3(mom), energy = std::sqrt(p * p + MUON_MASS * MUON_MASS);
    const float dt = h / (p / energy * C_LIGHT), q = charge * E_CHARGE;
    float k1[6], k2[6], k3[6], k4[6], t[6], B0[3], B1[3], Ba[3];
    field_at(e, pos[0], pos[1], pos[2], B0);
    deriv(s, q, B0, energy, k1);
    field_at(e, pos[0] + dt * k1[0], pos[1] + dt * k1[1], pos[2] + dt * k1[2], B1);
    for (int i = 0; i < 3; ++i) Ba[i] = 0.5f * (B0[i] + B1[i]);
    for (int i = 0; i < 6; ++i) t[i] = s[i] + 0.5f * dt * k1[i];
    deriv(t, q, Ba, energy, k2);
    for (int i = 0; i < 6; ++i) t[i] = s[i] + 0.5f * dt * k2[i];
    deriv(t, q, Ba, energy, k3);
    for (int i = 0; i < 6; ++i) t[i] = s[i] + dt * k3[i];
    deriv(t, q, Ba, energy, k4);
    for (int i = 0; i < 6; ++i) s[i] += dt / 6.0f * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
    for (int i = 0; i < 3; ++i) { pos[i] = s[i]; mom[i] = s[3 + i]; }
}

void propagate_one(const Env &e, float *P, float *M, float charge, float plane_z, float kill_at, int num_steps,
                   float step_fixed, uint64_t seed, uint64_t idx) {
    Rng rng(seed, idx);
    float pos[3] = {P[0], P[1], P[2]}, mom[3] = {M[0], M[1], M[2]}, out[3];
    const float *c = e.cav;
    for (int step = 0; step < num_steps; ++step) {
        const float p = norm3(mom);
        if (p < kill_at) break;
        if (std::fabs(pos[0]) > 10.0f || std::fabs(pos[1]) > 10.0f) break;
        const float u[3] = {mom[0] / p, mom[1] / p, mom[2] / p};
        const float pc = std::fmin(400.0f, std::fmax(0.18f, p));
        const int hb = int((std::log10(pc) - e.log_start) * e.inv_log_step);
        const float r = rng.uniform();
        const Material mat = material(e, pos[0], pos[1], pos[2]);
        float d1 = 0, d2 = 0;
        if (mat != AIR) {
            const Hist &h = e.mat[mat];
            const int tb = int(rng.next() % uint32_t(e.H));
            const int bin = r < h.prob[tb + hb * e.H] ? tb : h.alias[tb + hb * e.H];
            const float v1 = h.c1[bin] + (rng.uniform() - 0.5f) * h.w1[bin];
            const float v2 = h.c2[bin] + (rng.uniform() - 0.5f) * h.w2[bin];
            d1 = p * std::exp(v1);
            d2 = p * std::exp(v2);
        }
        const float phi = rng.uniform() * 2.0f * float(M_PI);
        const float dP[3] = {d2 * std::cos(phi), -d2 * std::sin(phi), -d1};
        rotate_vector(u, dP, mom, out);
        mom[0] = out[0]; mom[1] = out[1]; mom[2] = out[2];
        if (mom[2] < 0.0f) break;
        const bool open = pos[0] >= c[5] + BIG_STEP && pos[0] <= c[6] - BIG_STEP && pos[1] >= c[7] + BIG_STEP &&
                          pos[1] <= c[8] - BIG_STEP && pos[2] >= e.gm.z_max && pos[2] >= e.fm.endZ &&
                          pos[2] <= plane_z - BIG_STEP;
        rk4_cached(e, pos, mom, charge, open ? BIG_STEP : step_fixed);
        if (plane_z >= 0 && pos[2] >= plane_z) break;
    }
    for (int i = 0; i < 3; ++i) { P[i] = pos[i]; M[i] = mom[i]; }
}

void propagate_muons_with_alias_sampling(torch::Tensor positions, torch::Tensor momenta, torch::Tensor charges,
                                         py::dict histograms, torch::Tensor arb8s, torch::Tensor cells,
                                         torch::Tensor items, torch::Tensor field, torch::Tensor field_range,
                                         torch::Tensor cavern, bool use_symmetry, float plane_z, float kill_at,
                                         int num_steps, float step_fixed, int seed) {
    for (auto *t : {&positions, &momenta, &charges, &arb8s, &field, &field_range, &cavern})
        TORCH_CHECK(t->device().is_cpu() && t->scalar_type() == torch::kFloat32 && t->is_contiguous(),
                    "expected contiguous float32 CPU tensors");
    TORCH_CHECK(cells.scalar_type() == torch::kInt32 && items.scalar_type() == torch::kInt32, "z grid: int32");
    TORCH_CHECK(field_range.size(1) == 6, "field range: (1, 6)");
    Env e;
    std::vector<torch::Tensor> keep;
    const char *names[2] = {"iron", "concrete"};
    for (int m = 0; m < 2; ++m) {
        auto l = py::cast<py::list>(histograms[py::str(names[m])]);
        TORCH_CHECK(l.size() == 6, std::string("6 histogram tensors expected for ") + names[m]);
        std::vector<torch::Tensor> t;
        for (int i = 0; i < 6; ++i) t.push_back(py::cast<torch::Tensor>(l[i]).contiguous().cpu());
        t[1] = t[1].to(torch::kInt32);
        for (int i : {0, 2, 3, 4, 5}) t[i] = t[i].to(torch::kFloat32);
        e.mat[m] = Hist{t[0].data_ptr<float>(), t[2].data_ptr<float>(), t[3].data_ptr<float>(),
                        t[4].data_ptr<float>(), t[5].data_ptr<float>(), t[1].data_ptr<int>()};
        if (m == 0) {
            e.H = int(t[0].size(1));
            e.log_start = std::log10(0.18f);
            e.inv_log_step = float(t[0].size(0)) / (std::log10(400.0f) - e.log_start);
        }
        keep.insert(keep.end(), t.begin(), t.end());
    }
    const float *r = field_range.data_ptr<float>();
    FieldMeta &fm = e.fm;
    fm.binsX = int(field.size(0)); fm.binsY = int(field.size(1)); fm.binsZ = int(field.size(2));
    fm.startX = r[0]; fm.endX = r[1]; fm.startY = r[2]; fm.endY = r[3]; fm.startZ = r[4]; fm.endZ = r[5];
    fm.invX = 1.0f / (fm.endX - fm.startX); fm.invY = 1.0f / (fm.endY - fm.startY); fm.invZ = 1.0f / (fm.endZ - fm.startZ);
    fm.stride_y = fm.binsX * fm.binsZ;
    e.field = field.data_ptr<float>();

    const int n_arb = int(arb8s.size(0));
    std::vector<Arb8> blocks(n_arb);
    float z_min = 0.0f, z_max = 30.0f;
    if (n_arb > 0) {
        auto a = arb8s.accessor<float, 3>();
        z_min = 1e30f; z_max = -1e30f;
        for (int i = 0; i < n_arb; ++i) {
            for (int j = 0; j < 4; ++j)
                for (int c = 0; c < 2; ++c) { blocks[i].neg[j][c] = a[i][j][c]; blocks[i].pos[j][c] = a[i][j + 4][c]; }
            const float zn = a[i][0][2], zp = a[i][4][2];
            blocks[i].z_center = 0.5f * (zn + zp);
            blocks[i].dz = 0.5f * std::fabs(zp - zn);
            z_min = std::fmin(z_min, std::fmin(zn, zp));
            z_max = std::fmax(z_max, std::fmax(zn, zp));
        }
        z_max = std::fmax(z_max, 30.0f);
    }
    e.arb8s = blocks.data();
    e.cells = cells.data_ptr<int>();
    e.items = items.data_ptr<int>();
    e.gm.sz = int(cells.size(0)) - 1;
    e.gm.z_min = z_min; e.gm.z_max = z_max;
    e.gm.inv_h = float(e.gm.sz) / (z_max - z_min > 0 ? z_max - z_min : 1.0f);
    e.cav = cavern.data_ptr<float>();
    e.sym = use_symmetry;

    const int64_t N = positions.size(0);
    float *P = positions.data_ptr<float>(), *M = momenta.data_ptr<float>();
    const float *Q = charges.data_ptr<float>();
    {
        py::gil_scoped_release nogil;
#pragma omp parallel for schedule(dynamic, 4096)
        for (int64_t i = 0; i < N; ++i)
            propagate_one(e, P + 3 * i, M + 3 * i, Q[i], plane_z, kill_at, num_steps, step_fixed, uint64_t(seed), uint64_t(i));
    }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("propagate_muons_with_alias_sampling", &propagate_muons_with_alias_sampling,
          "CPU (OpenMP) port of faster_muons_torch.propagate_muons_with_alias_sampling (field map)");
    m.def("num_threads", []() { return omp_get_max_threads(); });
}
