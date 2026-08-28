TEST_CASES = [
    {
        "repo_name": "sampleproject",
        "question": "what encoding should the readme file use",
        "expected_tool": "code_search_tool",
        "expected_keywords": ["UTF-8"]
    },
    {
        "repo_name": "sampleproject",
        "question": "what calls build_and_check_dists",
        "expected_tool": "graph_search_tool",
        "expected_keywords": ["tests"]
    },
    {
        "repo_name": "To-Do-list",
        "question": "in which file are all the models present",
        "expected_tool": "code_search_tool",
        "expected_keywords": ["Task.java", "Model"]
    },
    {
        "repo_name": "To-Do-list",
        "question": "what calls findAll",
        "expected_tool": "graph_search_tool",
        "expected_keywords": ["TaskServiceImp.java", "getAllTask"]
    }
]
