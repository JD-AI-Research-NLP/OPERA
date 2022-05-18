# import json
#
# OP = ["addition", "substraction", "ZERO", "max", "min", "argmin", "argmax", "key_value", "count", "extract_span"]
#
# NUMBER_TRIGER = "how many"
#
# COUNT_TRIGER = [
#     "how many field goal",
#     "how many touchdown",
#     "how many TD"
#     "how many pass",
#     "how many win",
#     "how many loss",
#     "how many rushing",
#     "how many interceptions",
#     "how many quarters"
# ]
# DATE_DIFFRENCE_TRIGER = [
#     "how many year",
#     "how many month",
#     "how many day"
# ]
#
# DATE_COMPARE_TRIGER = [
#     "which happen"
#     "which event happen",
#     "which occur",
#     "which event occur"
# ]
#
# MAX_TRIGER = ["longest", "more"]
# MIN_TRIGER = ["shortest", "less"]
#
# ARGMAX_MIN_TRIGER = [
#     "who score",
#     "who kick",
#     "who caught"
#     "which player",
#     "which team",
# ]
#
# COMPARE_TRIGER = [
#     "between",
#     "and",
#     "over",
#     "compare",
#     "than",
#     "difference"
# ]
#
#
# def load_json_file(input_file_path):
#     with open(input_file_path, 'r') as f:
#         data = json.load(f)
#     f.close()
#     return data
#
#
# def question_parser(question_text):
#     '''
#     give tags for question to claim which ops are included in this question
#     '''
#     question_text = question_text.lower()
#     question_tag = []
#     if NUMBER_TRIGER in question_text:
#         if any(triger in question_text for triger in COUNT_TRIGER):
#             question_tag.append("count")
#             return question_tag
#         else:
#
#             if any(triger in question_text for triger in DATE_DIFFRENCE_TRIGER):
#                 question_tag.extend(["addition", "substraction"])
#                 return question_tag
#             else:
#
#                 if any(triger in question_text for triger in MAX_TRIGER):
#                     question_tag.extend(["max"])
#                     if any(triger in question_text for triger in COMPARE_TRIGER):
#                         question_tag.extend(["addition", "substraction"])
#
#                     question_tag = list(set(question_tag))
#                     return question_tag
#                 if any(triger in question_text for triger in MIN_TRIGER):
#                     question_tag.extend(["min"])
#                     if any(triger in question_text for triger in COMPARE_TRIGER):
#                         question_tag.extend(["addition", "substraction"])
#                     question_tag = list(set(question_tag))
#                     return question_tag
#
#                 question_tag = ["addition", "substraction"]
#                 return question_tag
#
#     elif any(triger in question_text for triger in ARGMAX_MIN_TRIGER):
#         question_tag.extend(['extract_span'])
#         if any(triger in question_text for triger in MAX_TRIGER):
#             question_tag.extend(['argmax', 'key_value'])
#             return question_tag
#         if any(triger in question_text for triger in MIN_TRIGER):
#             question_tag.extend(['argmin', 'key_value'])
#             return question_tag
#     else:
#         question_tag.extend(['extract_span'])
#         return question_tag
#
#     return question_tag
#
#
# if __name__ == '__main__':
#     # data_root = "../../numnet_plus_data/drop_dataset"
#     # train_file_name = "drop_dataset_train.json"
#     # train_file_data = os.path.join(data_root, train_file_name)
#
#     # text = "Which happened first, the defeat at Tangier, or Henry retiring to Sagres?"
#
#     text = "How many years did Henry live after his failure to capture Tangier?"
#     tag = question_parser(text)
#     print(tag)


import json

OP = ["addition", "substraction", "max", "min", "argmin", "argmax", "argmore", "argless", "key_value", "count",
      "extract_span"]

NUMBER_TRIGER = "how many"

COUNT_TRIGER = [
    "how many field goal",
    "how many touchdown",
    "how many TD"
    "how many pass",
    "how many win",
    "how many loss",
    "how many rushing",
    "how many interceptions",
    "how many quarters"
]
DATE_DIFFRENCE_TRIGER = [
    "how many year",
    "how many month",
    "how many day"
]

DATE_COMPARE_TRIGER = [
    "which happen"
    "which event happen",
    "which occur",
    "which event occur"
]

MAX_TRIGER = ["longest", "last", "highest"]
MIN_TRIGER = ["shortest", "first", 'lowest']

ARGMAX_MIN_TRIGER = [
    "who score",
    "who kick",
    "who caught",
    "who threw",
    "who had",
    "which player",
    "which team",
]

COMPARE_TRIGER = [
    "between",
    "and",
    "over",
    "compare",
    "than",
    "difference"
]

ARG_MORE_TRIGER = [
    "longer",
    "larger",
    "more",
]

ARG_LESS_TRIGER = [
    "shorter",
    "smaller",
    "less",
]


def load_json_file(input_file_path):
    with open(input_file_path, 'r') as f:
        data = json.load(f)
    f.close()
    return data


def question_parser(question_text):
    '''
    give tags for question to claim which ops are included in this question
    '''
    question_text = question_text.lower()
    question_tag = []
    if NUMBER_TRIGER in question_text:
        if any(triger in question_text for triger in COUNT_TRIGER):
            question_tag.append("count")
            return question_tag
        else:
            if any(triger in question_text for triger in DATE_DIFFRENCE_TRIGER):
                question_tag.extend(["addition", "substraction"])
                return question_tag
            else:
                if any(triger in question_text for triger in MAX_TRIGER):
                    question_tag.extend(["max"])
                    if any(triger in question_text for triger in COMPARE_TRIGER):
                        question_tag.extend(["addition", "substraction"])
                    question_tag = list(set(question_tag))
                if any(triger in question_text for triger in MIN_TRIGER):
                    question_tag.extend(["min"])
                    if any(triger in question_text for triger in COMPARE_TRIGER):
                        question_tag.extend(["addition", "substraction"])
                    question_tag = list(set(question_tag))
                if not any(triger in question_text for triger in MAX_TRIGER) and not any(
                        triger in question_text for triger in MIN_TRIGER):
                    question_tag = ["addition", "substraction"]
                return question_tag

    elif any(triger in question_text for triger in ARGMAX_MIN_TRIGER):
        if any(triger in question_text for triger in MAX_TRIGER):
            question_tag.extend(['argmax', 'key_value'])
            return question_tag
        if any(triger in question_text for triger in MIN_TRIGER):
            question_tag.extend(['argmin', 'key_value'])
            return question_tag

        if any(triger in question_text for triger in ARG_MORE_TRIGER):
            question_tag.extend(['argmore', 'key_value'])
            return question_tag
        if any(triger in question_text for triger in ARG_LESS_TRIGER):
            question_tag.extend(['argless', 'key_value'])
            return question_tag
        question_tag.extend(['extract_span'])

    else:
        question_tag.extend(['extract_span'])
        return question_tag

    return question_tag


if __name__ == '__main__':
    # text = "Which happened first, the defeat at Tangier, or Henry retiring to Sagres?

    # text = "How many years did Henry live after his failure to capture Tangier?"
    # tag = question_parser(text)
    # print(tag)
    import os

    # data_root = "../../numnet_plus_data/drop_dataset"
    # train_file_name = "generate_data/generate_data_dev.json"
    # train_file = os.path.join(data_root, train_file_name)
    # question_tag_file_name = "generate_data/generate_question_tag_dev.txt"
    # question_tag_file = os.path.join(data_root, question_tag_file_name)
    # question_tag_result = []
    #
    # with open(train_file, 'r') as f:
    #     train_data = json.load(f)
    # f.close()
    # for passage_id, passage_info in train_data.items():
    #     passage_text = passage_info["passage"]
    #     for question_answer in passage_info["qa_pairs"]:
    #         question_id = question_answer["query_id"]
    #         question_text = question_answer["question"].strip()
    #         question_tag = question_parser(question_text)
    #         question_tag_result.append(question_text + "\t" + str(question_tag) + "\n")
    #
    # with open(question_tag_file, 'w') as f:
    #     f.writelines(question_tag_result)
    # f.close()


    data_root = "../../numnet_plus_data/drop_dataset"
    train_file_name = "squad_data/drop_dataset_dev.json"
    train_file = os.path.join(data_root, train_file_name)
    question_tag_file_name = "squad_data/drop_dataset_tag_dev.txt"
    question_tag_file = os.path.join(data_root, question_tag_file_name)
    question_tag_result = []

    with open(train_file, 'r') as f:
        train_data = json.load(f)
    f.close()
    for passage_id, passage_info in train_data.items():
        passage_text = passage_info["passage"]
        for question_answer in passage_info["qa_pairs"]:
            question_id = question_answer["query_id"]
            question_text = question_answer["question"].strip()
            question_tag = question_parser(question_text)
            question_tag_result.append(question_text + "\t" + str(question_tag) + "\n")

    with open(question_tag_file, 'w') as f:
        f.writelines(question_tag_result)
    f.close()