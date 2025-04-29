import pickle

def unpickle_file(filename):
    try:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            print(f"Type of unpickled data: {type(data)}")
            print("Contents:")
            print(data)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    filename = input("Enter pickle file path: ")
    unpickle_file(filename)