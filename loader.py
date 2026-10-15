
import os
import tempfile

from langchain_community.document_loaders import TextLoader, PyPDFLoader
MAP_FOR_FILES = {
    "txt": TextLoader,
    "md": TextLoader,
    "py": TextLoader,
    "js": TextLoader,
    "java": TextLoader,
    "pdf": PyPDFLoader,
}


class load:
    @staticmethod
    def load_all_docs(files, map_for_files=MAP_FOR_FILES):
        all_docs = []

        for file in files:
            extension = file.name.split(".")[-1].lower()

            loader_class = map_for_files.get(extension)
            if loader_class is None:
                print(f"Skipping unsupported file: {file.name}")
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as temp:
                temp.write(file.getvalue())
                temp_path = temp.name

            try:
                docs = loader_class(temp_path).load()
                for d in docs:
                    d.metadata["source"] = file.name   
                all_docs.extend(docs)
            finally:
                os.remove(temp_path)                   

        return all_docs