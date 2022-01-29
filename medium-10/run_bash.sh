#!/bin/bash
set -x #echo on
#!/bin/bash -xv
PS4='${LINENO}: '
python3 src/action_masking_with_server_limit/proposed_method/run_this.py
python3 src/action_prioritization_based_on_server_group/proposed_method/run_this.py
python3 src/execution_and_action_prioritization_methods/baseline/run_this.py
python3 src/execution_and_action_prioritization_methods/baseline/run_this.py
python3 src/execution_and_action_prioritization_methods/baseline/run_this.py
python3 src/execution_and_action_prioritization_methods/baseline/run_this.py
python3 src/execution_prioritization_based_on_task_latency/proposed_method/run_this.py
python3 src/execution_prioritization_based_on_task_latency/proposed_method/run_this.py
python3 src/execution_prioritization_based_on_task_priority/proposed_method/run_this.py
