DEFAULT_TEMPLATE = '{instruction}\n{examples}\n{post_prompt}\n\n<context>\n{context}\n</context>\nQuestion: {question}'


def get_formatted_input(context, question, examples, instruction, post_prompt, template=DEFAULT_TEMPLATE):
    # pre_prompt - general instruction
    # examples - in-context examples
    # post_prompt - any additional instructions after examples
    # context - text to use for qa
    # question - question to answer based on context
    formatted_input = template.format(instruction=instruction, examples=examples, post_prompt=post_prompt,
                                      context=context.strip(), question=question)
    return formatted_input.strip()


DEFAULT_PROMPTS = {
    'qa1': {
        'instruction':
            'I will give you context with the facts about positions of different persons hidden in some random text '
            'and a question. You need to answer the question based only on the information from the facts. '
            'If a person was in different locations, use the latest location to answer the question.',
        'examples':
            '<example>\n'
            'Charlie went to the hallway. Judith come back to the kitchen. Charlie travelled to balcony. '
            'Where is Charlie?\n'
            'Answer: The most recent location of Charlie is balcony.\n'
            '</example>\n\n'
            '<example>\n'
            'Alan moved to the garage. Charlie went to the beach. Alan went to the shop. Rouse '
            'travelled to balcony. Where is Alan?\n'
            'Answer: The most recent location of Alan is shop.\n'
            '</example>',
        'post_prompt':
            'Always return your answer in the following format: '
            'The most recent location of ’person’ is ’location’. Do not write anything else after that. '
    },
    'qa2': {
        'instruction':
            'I give you context with the facts about locations and actions of different persons '
            'hidden in some random text and a question.'
            'You need to answer the question based only on the information from the facts.\n'
            'If a person got an item in the first location and travelled to the second location '
            'the item is also in the second location. '
            'If a person dropped an item in the first location and moved to the second location '
            'the item remains in the first location.',
        'examples':
            '<example>\n'
            'Charlie went to the kitchen. Charlie got a bottle. Charlie moved to the balcony. '
            'Where is the bottle?\n'
            'Answer: The bottle is in the balcony.\n'
            '</example>\n'
            '<example>\n'
            'Alan moved to the garage. Alan got a screw driver. Alan moved to the kitchen. Where '
            'is the screw driver?\n'
            'Answer: The screw driver is in the kitchen.\n'
            '</example>',
        'post_prompt':
            'Always return your answer in the following format: The ’item’ is in ’location’. '
            'Do not write anything else after that.'
    },
    'qa3': {
        'instruction':
            'I give you context with the facts about locations and actions of different persons '
            'hidden in some random text and a question. '
            'You need to answer the question based only on the information from the facts.\n'
            'If a person got an item in the first location and travelled to the second location '
            'the item is also in the second location. '
            'If a person dropped an item in the first location and moved to the second location '
            'the item remains in the first location',
        'examples':
            '<example>\n'
            'John journeyed to the bedroom.Mary grabbed the apple. Mary went back to the bathroom. '
            'Daniel journeyed to the bedroom. Daniel moved to the garden. Mary travelled to the kitchen. '
            'Where was the apple before the kitchen?\n'
            'Answer: Before the kitchen the apple was in the bathroom.\n'
            '</example>\n'
            '<example>\n'
            'John went back to the bedroom. John went back to the garden. John went back to the kitchen. '
            'Sandra took the football. Sandra travelled to the garden. Sandra journeyed to the bedroom. '
            'Where was the football before the bedroom?\n'
            'Answer: Before the kitchen the football was in the garden.\n'
            '</example>',
        'post_prompt':
            'Always return your answer in the following format: '
            'Before the $location_1& the $item$ was in the $location_2$. Do not write anything else after that.'
    },
    'qa4': {
        'instruction':
            'I will give you context with the facts about different people, their location and actions, hidden in '
            'some random text and a question. '
            'You need to answer the question based only on the information from the facts.',
        'examples':
            '<example>\n'
            'The hallway is south of the kitchen. The bedroom is north of the kitchen. '
            'What is the kitchen south of?\n'
            'Answer: bedroom\n'
            '</example>\n'
            '<example>\n'
            'The garden is west of the bedroom. The bedroom is west of the kitchen. What is west of the bedroom?\n'
            'Answer: garden\n'
            '</example>',
        'post_prompt':
            'Your answer should contain only one word - location. Do not write anything else after that.'
    },
    'qa5': {
        'instruction':
            'I will give you context with the facts about locations and their relations hidden in some random text '
            'and a question. You need to answer the question based only on the information from the facts.',
        'examples':
            '<example>\n'
            'Mary picked up the apple there. Mary gave the apple to Fred. Mary moved to the bedroom. '
            'Bill took the milk there. Who did Mary give the apple to?\n'
            'Answer: Fred\n'
            '</example>\n'
            '<example>\n'
            'Jeff took the football there. Jeff passed the football to Fred. Jeff got the milk there. '
            'Bill travelled to the bedroom. Who gave the football?\n'
            'Answer: Jeff\n'
            '</example>\n'
            '<example>\n'
            'Fred picked up the apple there. Fred handed the apple to Bill. Bill journeyed to the bedroom. '
            'Jeff went back to the garden. What did Fred give to Bill?\n'
            'Answer: apple\n'
            '</example>',
        'post_prompt':
            'Your answer should contain only one word. Do not write anything else after that. '
            'Do not explain your answer.'
    },
    'qa6': {
        'instruction':
            'I will give you context with the facts about people and their locations hidden in some random text and a '
            'question. You need to answer the question based only on the information from the facts. '
            'If a person was in different locations, use the latest location the person was in to answer the question.',
        'examples':
            '<example>\n'
            'John travelled to the hallway. John travelled to the garden. Is John in the garden?\n'
            'Answer: yes\n'
            '</example>\n'
            '<example>\n'
            'Mary went to the office. Daniel journeyed to the hallway. Mary went to the bedroom. '
            'Sandra went to the garden. Is Mary in the office?\n'
            'Answer: no\n'
            '</example>\n',
        'post_prompt':
            'Your answer should contain only one word - $yes$ or $no$. Do not write anything else after that. '
            'Do not explain your answer.'
    },
    'qa7': {
        'instruction':
            'I will give you context with the facts about people and objects they carry, hidden in some random text '
            'and a question. You need to answer the question based only on the information from the facts.',
        'examples':
            '<example>\n'
            'Daniel went to the bedroom. Daniel got the apple there. How many objects is Daniel carrying?\n'
            'Answer: one\n'
            '</example>\n'
            '<example>\n'
            'Mary grabbed the apple there. Mary gave the apple to John. How many objects is Mary carrying?\n'
            'Answer: none\n'
            '</example>\n'
            '<example>\n'
            'Sandra travelled to the hallway. Sandra picked up the milk there. Sandra took the apple there. '
            'Mary travelled to the garden. How many objects is Sandra carrying?\n'
            'Answer: two\n'
            '</example>\n',
        'post_prompt':
            'Your answer should contain only one word - $none$ or $number_of_objects$. '
            'Do not write anything else after that. Do not explain your answer.',
    },
    'qa8': {
        'instruction':
            'I will give you context with the facts about people and objects they carry, hidden in some random text '
            'and a question. You need to answer the question based only on the information from the facts.',
        'examples':
            '<example>\n'
            'Sandra travelled to the garden. Mary grabbed the milk there. What is Mary carrying?\n'
            'Answer: milk\n'
            '</example>\n'
            '<example>\n'
            'Mary travelled to the kitchen. Sandra travelled to the office. John travelled to the office. '
            'Sandra discarded the milk there. What is Sandra carrying?\n'
            'Answer: nothing\n'
            '</example>\n'
            '<example>\n'
            'Daniel grabbed the apple there. Mary went to the office. Daniel moved to the garden. '
            'Daniel grabbed the milk there. Mary went to the kitchen. What is Daniel carrying?\n'
            "Answer: apple,milk\n"
            "</example>\n",
        'post_prompt':
            'Your answer should contain only one or two words: $nothing$ or $object$ or $object_1$, $object_2$. '
            'Do not write anything else. Do not explain your answer.'
    },
    'qa9': {
        'instruction':
            'I will give you context with the facts about people and their locations hidden in some random text and '
            'a question. You need to answer the question based only on the information from the facts. '
            'If a person was in different locations, use the latest location the person was in to answer the question.',
        'examples':
            '<example>\n'
            'John is not in the bathroom. Sandra is not in the bedroom. Is John in the bathroom?\n'
            'Answer: no\n'
            '</example>\n'
            '<example>\n'
            'Mary journeyed to the kitchen. John is in the bedroom. Sandra is not in the garden. '
            'Is Mary in the kitchen?\n'
            'Answer: yes\n'
            '</example>\n',
        'post_prompt':
            'Your answer should contain only one word - $yes$ or $no$. Do not write anything else. '
            'Do not explain your answer.'
    },
    'qa10': {
        'instruction':
            'I will give you context with the facts about people and their locations hidden in some random text and a '
            'question. You need to answer the question based only on the information from the facts. '
            'If a person was in different locations, use the latest location the person was in to answer the question.',
        'examples':
            '<example>\n'
            'Bill is in the kitchen. Julie is either in the school or the cinema. Is Bill in the bedroom?\n'
            'Answer: no\n'
            '</example>\n'
            '<example>\n'
            'Fred is in the bedroom. Mary is either in the school or the cinema. Is Mary in the school?\n'
            'Answer: maybe\n'
            '</example>\n'
            '<example>\n'
            'Fred is either in the kitchen or the park. Bill moved to the cinema. Is Bill in the cinema?\n'
            'Answer: yes\n'
            '</example>\n'
            '<context>\n',
        'post_prompt':
            'Your answer should contain only one word - $yes$ or $no$ or $maybe$. Do not write anything else. '
            'Do not explain your answer.'
    }
}
