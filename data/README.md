### Generating BABILong samples

First, uncompress the bAbI dataset.
```bash
unzip tasks_1-20_v1-2.zip
```

To generate dataset samples, run:

```bash
python create_tasks.py "task_name1 task_name2 task_name3"
```
You can choose task names from 20 bAbI tasks:

`qa1_single-supporting-fact qa2_two-supporting-facts qa3_three-supporting-facts qa4_two-arg-relations qa5_three-arg-relations qa6_yes-no-questions qa7_counting qa8_lists-sets qa9_simple-negation qa10_indefinite-knowledge qa11_basic-coreference qa12_conjunction qa13_compound-coreference qa14_time-reasoning qa15_basic-deduction qa16_basic-induction qa17_positional-reasoning qa18_size-reasoning qa19_path-finding qa20_agents-motivations`

The number of samples and the path for the results folder can be changed in the beginning of `create_tasks.py`.