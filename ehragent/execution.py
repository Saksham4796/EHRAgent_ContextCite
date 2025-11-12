def execute_with_memory(user_proxy, chatbot, question, answer, selected_memory):
    # Execute task with given memory records and return logs
    user_proxy.update_memory(len(selected_memory), selected_memory)

    user_proxy.initiate_chat(
        chatbot,
        message=question,
    )

    logs = user_proxy._oai_messages

    logs_string = []
    logs_string.append(str(question))
    logs_string.append(str(answer))
    for agent in list(logs.keys()):
        for j in range(len(logs[agent])):
            if logs[agent][j]['content'] != None:
                logs_string.append(logs[agent][j]['content'])
            else:
                argnums = logs[agent][j]['function_call']['arguments']
                if type(argnums) == dict and 'cell' in argnums.keys():
                    logs_string.append(argnums['cell'])
                else:
                    logs_string.append(argnums)

    return logs_string