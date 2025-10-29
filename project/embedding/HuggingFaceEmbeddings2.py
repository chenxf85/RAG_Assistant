from langchain_huggingface import HuggingFaceEmbeddings

class HuggingFaceEmbeddings2:
    def __init__(self, model_name,model_kwargs,encode_kwargs):
        """
        model_name: 模型路径或huggingface模型名
        device: 'cuda' 或 'cpu'
        load_in_8bit: 是否8bit量化，节省显存
        """
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs

        self.model = None

    def _load_model(self):
        if self.model is None:
         #   print("🔄 Loading HuggingFace embedding model...")
            self.model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=self.model_kwargs,
                encode_kwargs=self.encode_kwargs
            )

    def __call__(self, texts):
        self._load_model()
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        self._load_model()
        return self.model.embed_query(text)
    def embed_documents(self,documents):
        self._load_model()
        return self.model.embed_documents(documents)