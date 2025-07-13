from document import DocumentProcessor
from pprint import pprint


def main(): 
    dp = DocumentProcessor()

    documents = dp.load_documents()
    documents = dp.split_documents(documents)
    for i in documents: 
        print(i)
        print("\n\n\n")

if __name__ == "__main__": 
    main()