C     Forward hadrons of 400 GeV proton-nucleon collisions with PYTHIA 6.4.28 (SHiP fork v6.4.28-ship1),
C     the Pythia-6 alternative to pythia_mesons.cc (same output format), for generator comparisons.
C
C     pythia6_mesons N_EVENTS SEED EMIN OUT.bin [Z_OVER_A]
C
C     Fixed target ('FIXT'), beam p+ 400 GeV on p+ (fraction Z/A) then on n0 (Pythia 6 keeps one setup at a
C     time, so the two targets are generated one after the other). Inelastic processes: hard QCD (11, 12,
C     13, 28, 53, 68), low-pT (95), single and double diffraction (92, 93, 94); no elastic. Defaults
C     otherwise. Particles with c tau > 10 mm stay stable (MSTJ(22)=2, PARJ(71)=10), as FairShip leaves their
C     decays to Geant4. Writes every final-state hadron (no leptons, photons) with E > EMIN as float64 records
C     (event, pdg, px, py, pz, E); OUT.bin.txt: counts and settings.
      PROGRAM P6MESONS
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      INTEGER PYK, PYCHGE, PYCOMP
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYSUBS/MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON/PYDATR/MRPY(6),RRPY(100)
      COMMON/PYDAT3/MDCY(500,3),MDME(8000,2),BRAT(8000),KFDP(8000,5)
      CHARACTER*256 ARG, OUTF
      CHARACTER*4 TGT
      INTEGER*8 NEV, NP, NDONE, NWRIT, IEV, NTARG
      DOUBLE PRECISION REC(6), SIGP, SIGN
      INTEGER ISEED, IT, I, KA, KVM(5), JV
      LOGICAL STABVM, CHARM
      DATA KVM /221, 223, 113, 331, 333/
      CALL GET_COMMAND_ARGUMENT(1, ARG)
      READ(ARG,*) NEV
      CALL GET_COMMAND_ARGUMENT(2, ARG)
      READ(ARG,*) ISEED
      CALL GET_COMMAND_ARGUMENT(3, ARG)
      READ(ARG,*) EMIN
      CALL GET_COMMAND_ARGUMENT(4, OUTF)
      ZA = 74D0/183.84D0
      IF (COMMAND_ARGUMENT_COUNT() .GE. 5) THEN
        CALL GET_COMMAND_ARGUMENT(5, ARG)
        READ(ARG,*) ZA
      ENDIF
C     6th argument: 'vm' = eta/omega/rho0/eta'/phi stable (forced dimuons in Geant4); 'charm' = forced
C     c cbar (MSEL=4) with FairShip's makeCascade.py PoorE791 tune, charm and tau decayed by Pythia, and only
C     MUONS written (E > EMIN)
      STABVM = .FALSE.
      CHARM = .FALSE.
      IF (COMMAND_ARGUMENT_COUNT() .GE. 6) THEN
        CALL GET_COMMAND_ARGUMENT(6, ARG)
        STABVM = TRIM(ARG) .EQ. 'vm'
        CHARM = TRIM(ARG) .EQ. 'charm'
      ENDIF
      NP = NINT(ZA*NEV, 8)
      OPEN(21, FILE=TRIM(OUTF), ACCESS='STREAM', FORM='UNFORMATTED', STATUS='REPLACE')
      NDONE = 0
      NWRIT = 0
      SIGP = 0D0
      SIGN = 0D0
      DO IT = 1, 2
        IF (IT .EQ. 1) THEN
          TGT = 'p+'
          NTARG = NP
        ELSE
          TGT = 'n0'
          NTARG = NEV - NP
        ENDIF
        MRPY(1) = 19780503 + 2*ISEED + IT
        MRPY(2) = 0
        MSEL = 0
        DO I = 1, 500
          MSUB(I) = 0
        ENDDO
C       (inelastic minimum bias below; MSEL is set to 4 afterwards in charm mode, which overrides MSUB)
        MSUB(11) = 1
        MSUB(12) = 1
        MSUB(13) = 1
        MSUB(28) = 1
        MSUB(53) = 1
        MSUB(68) = 1
        MSUB(92) = 1
        MSUB(93) = 1
        MSUB(94) = 1
        MSUB(95) = 1
        MSTJ(22) = 2
        PARJ(71) = 10D0
        MSTP(122) = 0
        MSTU(12) = 12345
        MSTU(13) = 0
        IF (CHARM) THEN
          MSEL = 4
          PARP(2) = 2D0
          PARP(91) = 1.6D0
          MSTP(82) = 1
          PARP(89) = 1000D0
        ENDIF
        IF (STABVM) THEN
          DO JV = 1, 5
            MDCY(PYCOMP(KVM(JV)),1) = 0
          ENDDO
        ENDIF
        CALL PYINIT('FIXT', 'p+', TGT, 400D0)
        DO IEV = 1, NTARG
          CALL PYEVNT
          DO I = 1, N
            IF (K(I,1) .LT. 1 .OR. K(I,1) .GT. 10) CYCLE
            KA = ABS(K(I,2))
            IF (CHARM) THEN
              IF (KA .NE. 13) CYCLE
              GOTO 50
            ENDIF
            IF (KA .EQ. 11 .OR. KA .EQ. 12 .OR. KA .EQ. 13 .OR. KA .EQ. 14
     &          .OR. KA .EQ. 16) CYCLE
   50       IF (P(I,4) .LT. EMIN) CYCLE
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
        CALL PYSTAT(0)
        IF (IT .EQ. 1) SIGP = PARI(1)
        IF (IT .EQ. 2) SIGN = PARI(1)
        NDONE = NDONE + NTARG
      ENDDO
      CLOSE(21)
      OPEN(22, FILE=TRIM(OUTF)//'.txt', STATUS='REPLACE')
      WRITE(22,'(A,I12)') 'events_requested ', NEV
      WRITE(22,'(A,I12)') 'events_generated ', NDONE
      WRITE(22,'(A,I12)') 'events_on_proton ', NP
      WRITE(22,'(A,I12)') 'mesons_written ', NWRIT
      WRITE(22,'(A,F10.3)') 'emin ', EMIN
      WRITE(22,'(A,F10.6)') 'z_over_a ', ZA
      WRITE(22,'(A,I12)') 'seed ', ISEED
      WRITE(22,'(A,F12.4)') 'sigma_pp_mb ', SIGP
      WRITE(22,'(A,F12.4)') 'sigma_pn_mb ', SIGN
      WRITE(22,'(A)') 'generator pythia-6.4.28 (ShipSoft v6.4.28-ship1)'
      CLOSE(22)
      WRITE(*,'(I12,A,I12,A)') NDONE, ' events, ', NWRIT, ' hadrons written'
      END
