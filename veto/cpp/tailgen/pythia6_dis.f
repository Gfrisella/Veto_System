C     Lepton-nucleon interaction library with PYTHIA 6.4.28: final states of muon (or neutrino) beams on p+ / n0.
C
C     pythia6_dis BEAM TARGET E NEV SEED OUT.bin [LOG] [SETTING=VALUE ...]   (settings via PYGIVE)
C         BEAM    Pythia beam name: gamma/mu- or gamma/mu+ (muon-nucleon through the photon flux: real and
C                 virtual photons, VMD + direct + anomalous + DIS, MSTP(14)=30 default), mu- / mu+ (DIS only,
C                 as FairShip's muonDIS/makeMuonDIS.py), nu_mu / nu_mubar (neutrino DIS, W / Z exchange)
C         TARGET  p+ or n0, at rest; the beam along +z with energy E [GeV]
C         Writes every final-state particle (K(I,1) = 1) as float64 records
C             (event, pdg, px, py, pz, E, vx, vy, vz)   momenta GeV, production vertex in m from the interaction;
C         Pythia's default decays (K_S, Lambda, Sigma, Xi, Omega, pi0, ...; pi+-, K+-, K_L, n stable).
C         OUT.bin.txt: events, cross section PARI(1) [mb], failures (rewritten every 1000 events).
      PROGRAM P6DIS
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYSUBS/MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON/PYDATR/MRPY(6),RRPY(100)
      CHARACTER*256 ARG, OUTF, LOGF
      CHARACTER*16 BEAM, TGT
      INTEGER*8 NEV, IEV, NWRIT
      INTEGER ISEED, I, NFAIL
      DOUBLE PRECISION REC(9)
      CALL GET_COMMAND_ARGUMENT(1, ARG)
      BEAM = ARG(1:16)
      CALL GET_COMMAND_ARGUMENT(2, ARG)
      TGT = ARG(1:16)
      CALL GET_COMMAND_ARGUMENT(3, ARG)
      READ(ARG,*) EBEAM
      CALL GET_COMMAND_ARGUMENT(4, ARG)
      READ(ARG,*) NEV
      CALL GET_COMMAND_ARGUMENT(5, ARG)
      READ(ARG,*) ISEED
      CALL GET_COMMAND_ARGUMENT(6, OUTF)
      MSTU(11) = 11
      LOGF = '/dev/null'
      DO I = 7, COMMAND_ARGUMENT_COUNT()
        CALL GET_COMMAND_ARGUMENT(I, ARG)
        IF (INDEX(ARG, '=') .EQ. 0) LOGF = ARG
      ENDDO
      OPEN(11, FILE=TRIM(LOGF))
      MSTU(22) = 100000
      MSTP(122) = 0
C     minimum CM energy 2 GeV (default 10 GeV), as FairShip makeMuonDIS.py / makeCascade.py
      PARP(2) = 2D0
C     extra settings: arguments with '=' go to PYGIVE, e.g. 'CKIN(1)=1.0'
      DO I = 7, COMMAND_ARGUMENT_COUNT()
        CALL GET_COMMAND_ARGUMENT(I, ARG)
        IF (INDEX(ARG, '=') .GT. 0) CALL PYGIVE(TRIM(ARG))
      ENDDO
      MRPY(1) = 19780503 + 2*ISEED
      MRPY(2) = 0
      CALL PYINIT('FIXT', TRIM(BEAM), TRIM(TGT), EBEAM)
      OPEN(21, FILE=TRIM(OUTF), ACCESS='STREAM', FORM='UNFORMATTED', STATUS='REPLACE')
      NWRIT = 0
      NFAIL = 0
      DO IEV = 1, NEV
        CALL PYEVNT
        IF (MSTU(24) .NE. 0) NFAIL = NFAIL + 1
        DO I = 1, N
          IF (K(I,1) .NE. 1) CYCLE
          REC(1) = DBLE(IEV - 1)
          REC(2) = DBLE(K(I,2))
          REC(3) = P(I,1)
          REC(4) = P(I,2)
          REC(5) = P(I,3)
          REC(6) = P(I,4)
          REC(7) = V(I,1) * 1D-3
          REC(8) = V(I,2) * 1D-3
          REC(9) = V(I,3) * 1D-3
          WRITE(21) REC
          NWRIT = NWRIT + 1
        ENDDO
C       complete events on disk (a rare endless PYEVNT loop is killed by the caller, see bin/dis_library.py)
        FLUSH(21)
        IF (MOD(IEV, 1000_8) .EQ. 0 .OR. IEV .EQ. NEV) CALL WRSTAT(OUTF, IEV, NFAIL, NWRIT)
      ENDDO
      CLOSE(21)
      END

      SUBROUTINE WRSTAT(OUTF, IEV, NFAIL, NWRIT)
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      CHARACTER*256 OUTF
      INTEGER*8 IEV, NWRIT
      INTEGER NFAIL
      CALL PYSTAT(0)
      OPEN(22, FILE=TRIM(OUTF)//'.txt', STATUS='REPLACE')
      WRITE(22,'(A,I12)') 'events ', IEV
      WRITE(22,'(A,ES16.8)') 'sigma_mb ', PARI(1)
      WRITE(22,'(A,I12)') 'failures ', NFAIL
      WRITE(22,'(A,I12)') 'particles ', NWRIT
      CLOSE(22)
      END
