import argparse as ap
def display(name):
    print("My name is " + name)

def Main():
    parser = ap.ArgumentParser()
    parser.add_argument("name", help="name of person", type=str)

    args = parser.parse_args()

    if args.name in ['True', 'TRUE', 'true']:
        print("This is displayed because of true value for name!!!")

    elif args.name in ['False', 'FALSE', 'false']:
        print("This is displayed because of false value for name!!!")

if __name__=='__main__':
    Main()