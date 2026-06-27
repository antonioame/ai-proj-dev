@echo off
cd /d "U:\AI-Partition\progetto_v2\ai_private_proj"
call conda activate ai_env
python scripts/launch_race.py --driver bc_model --laps 1
pause
