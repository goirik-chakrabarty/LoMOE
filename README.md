# LoMOE
LoMOE: Localized Multi-Object Editing via Multi-Diffusion

benchmark/
    methods/
        bld/
        diffedit/
        glide/
        instruct-p2p

    metrics/
        MOE/
        SOE/

data/
    LoMOE_Bench/
        images/
        masks/
        LoMOE.json

    LoSOE_Bench/
        images/
        masks/
        LoSOE.json

src/
    edit/
        utils.py
        main.py
        run.bash -> readme
    invert/
        assets/
        src/
        run_multi.py -> rename
        run.py  -> merge
        run2.py -> merge
        run.bash -> readme