- train:

```bash
python phc/run_hydra.py learning=nonamp env=env_amp_heading robot=smpl_humanoid env.motion_file=data/amass/amass_train_take6_upright.pkl exp_name=<todo>
```
- test:

```bash
python phc/run_hydra.py learning=amp env=env_amp_heading robot=smpl_humanoid robot.box_body=False env.num_envs=1 headless=False epoch=-1 env.motion_file=data/amass/amass_train_take6_upright.pkl test=True no_virtual_display=True exp_name=<todo>
```