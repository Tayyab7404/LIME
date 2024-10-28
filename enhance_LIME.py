import cv2
from LIME import LIME

def main():
    # load
    name = input("Enter the file name: ")
    filePath = f"./pics/{name}"

    # initiate LIME operator and load pictures
    lime = LIME(iterations=30, alpha=0.15, rho=1.1, gamma=0.6, strategy=2, exact=True)
    lime.load(filePath)

    # run
    R = lime.run()
    
    # save results
    # filename = os.path.split(filePath)[-1]
    # savePath = f"./pics/outputs/LIME_{filename}"

    savePath = f"./pics/outputs/LIME_{name}"
    cv2.imwrite(savePath, R)

if __name__ == "__main__":
    main()