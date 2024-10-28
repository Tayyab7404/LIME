from LIME import LIME
import cv2

class DUAL:
    # initiate LIME operators used in DUAL
    def __init__(self, iterations, alpha, rho, gamma, limestrategy):
        self.limecore = LIME(iterations, alpha, rho, gamma, limestrategy, exact=True)

    # load enhanced images
    def load(self, imgPath, name):
        self.img = cv2.imread(imgPath) / 255
        self.imgrev = 1 - self.img
        self.imgname = name

    # multi-exposure image fusion 
    def multi_exposureimageFushion(self):
        mergecore = cv2.createMergeMertens(1, 1, 1)
        img = mergecore.process([self.img, self.forwardimg, self.reverseimg])
        return img

    def run(self):
        # generate enhanced image
        print('Using LIME for forward illumination enhancement:')
        self.limecore.loadimage(self.img)
        self.forwardimg = self.limecore.run()
        cv2.imwrite(f"./pics/outputs/DUAL_forward_{self.imgname}", self.forwardimg)

        # generate suppressed image
        print('Using LIME for reverse illumination enhancement:')
        self.limecore.loadimage(self.imgrev)
        self.reverseimg = 255 - self.limecore.run()
        cv2.imwrite(f"./pics/outputs/DUAL_reverse_{self.imgname}", self.reverseimg)

        # fuse input image, enhanced image and suppressed image to generate final results
        print('Use multi-exposure image fusion to generate the result:')
        l = self.multi_exposureimageFushion()
        cv2.imwrite(f"./pics/outputs/DUAL_result_{self.imgname}", l * 255)