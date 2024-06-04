TASK_LABELS = {
    'qa1': {'bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'},
    'qa2': {'bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'},
    'qa3': {'bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'},
    'qa4': {'bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'},
    'qa5': {'Bill', 'Fred', 'Jeff', 'Mary', 'apple', 'football', 'milk'},
    'qa6': {'no', 'yes'},
    'qa7': {'two', 'one', 'none'},
    'qa9': {'no', 'yes'},
    'qa10': {'maybe', 'no', 'yes'}
    }


def preprocess_output(output):
    output = output.lower()
    # take only the first sentence from output
    output = output.split('.')[0]
    # filter responses when model tries to generate examples
    output = output.split('<context>')[0]
    output = output.split('<example>')[0]
    output = output.split('Question')[0]
    return output


def compare_answers(target, output, question, task_labels):
    output = preprocess_output(output)
    target = target.lower()
    task_labels = {label.lower() for label in task_labels}

    # extract labels that were mentioned in the question
    labels_in_question = {label for label in task_labels if label in question.lower()}
    # extract labels that were mentioned in the model output
    labels_in_output = {label for label in task_labels if label in output}
    # filter labels in the output to exclude mentioned in the question
    # mentions in questions are never targets
    labels_in_output = labels_in_output - labels_in_question

    # check if the target is the only prediction
    if target in labels_in_output and len(labels_in_output) == 1:
        return True

    return False
