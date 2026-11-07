#!/bin/bash
# partition for the next pcf80 sweep run. download is not usable for this sweep: its association limits are MaxSubmit 1, MaxJobs 1,
# cpu=8, mem=32G per user, and another session's job holds it (checked 10-29). debug-cpu: MaxSubmit 20, node=1 per user, 2 h.
echo debug-cpu
