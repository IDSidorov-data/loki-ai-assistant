class LokiOrchestrator:
    def __init__(self):
        # ... существующий код ...
        self.llm_client = OllamaLLMClient()  
        self._initialize_resources()
