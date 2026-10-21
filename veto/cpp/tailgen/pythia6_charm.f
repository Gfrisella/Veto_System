C     Cascade charm with PYTHIA 6.4.28 as FairShip's macro/makeCascade.py: forced c cbar (MSEL=4) with the
C     PoorE791 tune (PARP(2)=2, PARP(91)=1.6, MSTP(82)=1, PARP(89)=1000), fixed target, any cascade projectile.
C
C     pythia6_charm xsec BEAM MOM NEV SEED
C         prints  BEAM MOM TARGET SIGMA_CC SIGMA_TOT  (mb, MSEL=4 and MSEL=2, as makeCascade.py's chi tables)
C         for the targets p+ and n0
C     pythia6_charm gen BEAM MOM NEV SEED EMIN OUT.bin [Z_OVER_A]
C         NEV forced c cbar events of BEAM (Pythia name, e.g. p+ n0 pi+ pi- K+ K- KL0 KS0) at momentum MOM along +z
C         on p+ (fraction Z/A, default 0.43 as makeCascade.py's fracp) then n0; charm and taus decayed by Pythia;
C         writes muons with E > EMIN as float64 records (event, pdg, px, py, pz, E); OUT.bin.txt: counts.
      PROGRAM P6CHARM
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYSUBS/MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON/PYDATR/MRPY(6),RRPY(100)
      CHARACTER*256 ARG, OUTF
      CHARACTER*8 MODE, BEAM, TGT
      INTEGER*8 NEV, NP, NTARG, IEV, NDONE, NWRIT
      INTEGER ISEED, IT, I
      DOUBLE PRECISION REC(6), SIGCC, SIGTOT
      CALL GET_COMMAND_ARGUMENT(1, ARG)
      MODE = ARG(1:8)
      CALL GET_COMMAND_ARGUMENT(2, ARG)
      BEAM = ARG(1:8)
      CALL GET_COMMAND_ARGUMENT(3, ARG)
      READ(ARG,*) PMOM
      CALL GET_COMMAND_ARGUMENT(4, ARG)
      READ(ARG,*) NEV
      CALL GET_COMMAND_ARGUMENT(5, ARG)
      READ(ARG,*) ISEED
      MSTU(12) = 12345
      MSTU(13) = 0
      MSTU(11) = 11
      OPEN(11, FILE='/dev/null')
      PARP(2) = 2D0
      PARP(91) = 1.6D0
      MSTP(82) = 1
      PARP(89) = 1000D0
      MSTP(122) = 0
      IF (TRIM(MODE) .EQ. 'xsec') THEN
        DO IT = 1, 2
          IF (IT .EQ. 1) TGT = 'p+'
          IF (IT .EQ. 2) TGT = 'n0'
          MRPY(1) = 19780503 + 2*ISEED + IT
          MRPY(2) = 0
          MSEL = 4
          CALL PYINIT('FIXT', TRIM(BEAM), TRIM(TGT), PMOM)
          DO IEV = 1, NEV
            CALL PYEVNT
          ENDDO
          CALL PYSTAT(0)
          SIGCC = PARI(1)
          MSEL = 2
          CALL PYINIT('FIXT', TRIM(BEAM), TRIM(TGT), PMOM)
          DO IEV = 1, MAX(NEV/10, 100_8)
            CALL PYEVNT
          ENDDO
          CALL PYSTAT(0)
          SIGTOT = PARI(1)
          WRITE(*,'(A,1X,F8.2,1X,A,2(1X,ES14.6))') TRIM(BEAM), PMOM, TRIM(TGT), SIGCC, SIGTOT
        ENDDO
        STOP
      ENDIF
      CALL GET_COMMAND_ARGUMENT(6, ARG)
      READ(ARG,*) EMIN
      CALL GET_COMMAND_ARGUMENT(7, OUTF)
      ZA = 0.43D0
      IF (COMMAND_ARGUMENT_COUNT() .GE. 8) THEN
        CALL GET_COMMAND_ARGUMENT(8, ARG)
        READ(ARG,*) ZA
      ENDIF
      NP = NINT(ZA*NEV, 8)
      OPEN(21, FILE=TRIM(OUTF), ACCESS='STREAM', FORM='UNFORMATTED', STATUS='REPLACE')
      NDONE = 0
      NWRIT = 0
      DO IT = 1, 2
        IF (IT .EQ. 1) THEN
          TGT = 'p+'
          NTARG = NP
        ELSE
          TGT = 'n0'
          NTARG = NEV - NP
        ENDIF
        IF (NTARG .LE. 0) CYCLE
        MRPY(1) = 19780503 + 2*ISEED + IT
        MRPY(2) = 0
        MSEL = 4
        CALL PYINIT('FIXT', TRIM(BEAM), TRIM(TGT), PMOM)
        DO IEV = 1, NTARG
          CALL PYEVNT
          DO I = 1, N
            IF (K(I,1) .LT. 1 .OR. K(I,1) .GT. 10) CYCLE
            IF (ABS(K(I,2)) .NE. 13) CYCLE
            IF (P(I,4) .LT. EMIN) CYCLE
            REC(1) = DBLE(NDONE + IEV - 1)
            REC(2) = DBLE(K(I,2))
            REC(3) = P(I,1)
            REC(4) = P(I,2)
            REC(5) = P(I,3)
            REC(6) = P(I,4)
            WRITE(21) REC
            NWRIT = NWRIT + 1
          ENDDO
        ENDDO
        NDONE = NDONE + NTARG
      ENDDO
      CLOSE(21)
      OPEN(22, FILE=TRIM(OUTF)//'.txt', STATUS='REPLACE')
      WRITE(22,'(A,I12)') 'events_generated ', NDONE
      WRITE(22,'(A,I12)') 'muons_written ', NWRIT
      CLOSE(22)
      END
