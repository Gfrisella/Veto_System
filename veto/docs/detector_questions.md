# Questions for the detector colleagues (veto simulation, ISSUES C2 + C6)

Drafted 9 Nov 2026 for Guglielmo to send. Current values in the simulation are in brackets; they live in
`lib/veto/technology.py` (`MODULES`, `frame`, `frame_edges`, `sub_frame`, timing / efficiency entries).

## Dead frames and their material (C2)
1. Dead border per module edge, per technology: straw [3 cm: frame + manifolds], scintillator tile / tile2
   [1 cm, SiPM side], scintillating fibre [2 cm, readout end], pixel [1 mm]. Is it on all four edges or only on the
   readout / manifold side(s)?
2. Material of those frames (e.g. Al, carbon fibre, G10, steel; thickness): needed for scattering and DIS in the frames.
3. Module size: is the 50 cm grid (multiples of 0.5 m) realistic for every technology? Maximum single-module length
   (straw tube length, fibre mat length, tile plane size)?
4. Straw planes: overlap / gap between neighbouring 1 m straw rows, and the frame of the tensioned UBT straws inside
   the acceptance (SR-370 open question).

## Technology numbers (C6)
5. Fibre timing resolution [0.5 ns per plane]: can we assume ~100 ps (Ettore)? With which readout?
6. Fibre plane efficiency [99.9 %]: per SiPM / per channel efficiency and dead-channel fraction to model?
7. Tiles: channel count of a 1 x 2.7 m plane for a realistic mix of tile sizes [single size now]; time
   resolution and efficiency per tile.
8. Maximum rates: fibre channel [1 MHz], pixel [3.5e8 Hz/cm^2]: still the right limits? Straw and tile limits?
9. Correlated inefficiencies (whole module / readout board dead) that we should include?
