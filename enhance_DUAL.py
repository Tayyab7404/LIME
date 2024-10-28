import cv2
from DUAL import DUAL

def main():
    # load
    name = input("Enter the file name: ")
    filePath = f"./pics/{name}"

    # initiate DUAL operator and load pictures
    dual = DUAL(iterations=30, alpha=0.15, rho=1.1, gamma=0.6, limestrategy=2)
    dual.load(filePath, name)

    # run
    dual.run()

if __name__ == "__main__":
    main()