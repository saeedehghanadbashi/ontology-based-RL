#!/bin/bash
set -x #echo on
#!/bin/bash -xv
PS4='${LINENO}: '
python3 src/execution_prioritization_based_on_task_latency/proposed_method/run_this.py
python3 src/observation_masking/user_card_number/proposed_method/run_this.py
python3 src/observation_masking/user_card_number/proposed_method/run_this.py
