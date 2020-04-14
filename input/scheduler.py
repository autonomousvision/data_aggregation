from imgaug import augmenters as iaa
import numpy as np

def medium(image_iteration):

    iteration = image_iteration/(120*1.5)
    frequency_factor = 0.05 + float(iteration)/1000000.0
    color_factor = float(iteration)/1000000.0
    dropout_factor = 0.198667 + (0.03856658 - 0.198667) / (1 + (iteration / 196416.6) ** 1.863486)

    blur_factor = 0.5 + (0.5*iteration/100000.0)

    add_factor = 10 + 10*iteration/150000.0

    multiply_factor_pos = 1 + (2.5*iteration/500000.0)
    multiply_factor_neg = 1 - (0.91 * iteration / 500000.0)

    contrast_factor_pos = 1 + (0.5*iteration/500000.0)
    contrast_factor_neg = 1 - (0.5 * iteration / 500000.0)


    #print 'Augment Status ',frequency_factor,color_factor,dropout_factor,blur_factor,add_factor,\
    #    multiply_factor_pos,multiply_factor_neg,contrast_factor_pos,contrast_factor_neg


    augmenter = iaa.Sequential([

        iaa.Sometimes(frequency_factor, iaa.GaussianBlur((0, blur_factor))),
        # blur images with a sigma between 0 and 1.5
        iaa.Sometimes(frequency_factor, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0,dropout_factor ),
                                                                  per_channel=color_factor)),
        # add gaussian noise to images
        iaa.Sometimes(frequency_factor, iaa.CoarseDropout((0.0, dropout_factor), size_percent=(
            0.08, 0.2), per_channel=color_factor)),
        # randomly remove up to X% of the pixels
        iaa.Sometimes(frequency_factor, iaa.Dropout((0.0, dropout_factor), per_channel=color_factor)),
        # randomly remove up to X% of the pixels
        iaa.Sometimes(frequency_factor,
                      iaa.Add((-add_factor, add_factor), per_channel=color_factor)),
        # change brightness of images (by -X to Y of original value)
        iaa.Sometimes(frequency_factor,
                      iaa.Multiply((multiply_factor_neg, multiply_factor_pos), per_channel=color_factor)),
        # change brightness of images (X-Y% of original value)
        iaa.Sometimes(frequency_factor, iaa.ContrastNormalization((contrast_factor_neg, contrast_factor_pos),
                                                                       per_channel=color_factor)),
        # improve or worsen the contrast
        iaa.Sometimes(frequency_factor, iaa.Grayscale((0.0, 1))),  # put grayscale

    ],
        random_order=True  # do all of the above in random order
    )

    return augmenter


def soft(image_iteration):

    iteration = image_iteration/(120*1.5)
    frequency_factor = 0.05 + float(iteration)/1200000.0
    color_factor = float(iteration)/1200000.0
    dropout_factor = 0.198667 + (0.03856658 - 0.198667) / (1 + (iteration / 196416.6) ** 1.863486)

    blur_factor = 0.5 + (0.5*iteration/120000.0)

    add_factor = 10 + 10*iteration/170000.0

    multiply_factor_pos = 1 + (2.5*iteration/800000.0)
    multiply_factor_neg = 1 - (0.91 * iteration / 800000.0)

    contrast_factor_pos = 1 + (0.5*iteration/800000.0)
    contrast_factor_neg = 1 - (0.5 * iteration / 800000.0)


    #print ('iteration',iteration,'Augment Status ',frequency_factor,color_factor,dropout_factor,blur_factor,add_factor,
    #    multiply_factor_pos,multiply_factor_neg,contrast_factor_pos,contrast_factor_neg)


    augmenter = iaa.Sequential([

        iaa.Sometimes(frequency_factor, iaa.GaussianBlur((0, blur_factor))),
        # blur images with a sigma between 0 and 1.5
        iaa.Sometimes(frequency_factor, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0,dropout_factor ),
                                                                  per_channel=color_factor)),
        # add gaussian noise to images
        iaa.Sometimes(frequency_factor, iaa.CoarseDropout((0.0, dropout_factor), size_percent=(
            0.08, 0.2), per_channel=color_factor)),
        # randomly remove up to X% of the pixels
        iaa.Sometimes(frequency_factor, iaa.Dropout((0.0, dropout_factor), per_channel=color_factor)),
        # randomly remove up to X% of the pixels
        iaa.Sometimes(frequency_factor,
                      iaa.Add((-add_factor, add_factor), per_channel=color_factor)),
        # change brightness of images (by -X to Y of original value)
        iaa.Sometimes(frequency_factor,
                      iaa.Multiply((multiply_factor_neg, multiply_factor_pos), per_channel=color_factor)),
        # change brightness of images (X-Y% of original value)
        iaa.Sometimes(frequency_factor, iaa.ContrastNormalization((contrast_factor_neg, contrast_factor_pos),
                                                                       per_channel=color_factor)),
        # improve or worsen the contrast
        iaa.Sometimes(frequency_factor, iaa.Grayscale((0.0, 1))),  # put grayscale

    ],
        random_order=True  # do all of the above in random order
    )

    return augmenter


def high(image_iteration):

    iteration = image_iteration/(120*1.5)
    frequency_factor = 0.05 + float(iteration)/800000.0
    color_factor = float(iteration)/800000.0
    dropout_factor = 0.198667 + (0.03856658 - 0.198667) / (1 + (iteration / 196416.6) ** 1.863486)

    blur_factor = 0.5 + (0.5*iteration/80000.0)

    add_factor = 10 + 10*iteration/120000.0

    multiply_factor_pos = 1 + (2.5*iteration/350000.0)
    multiply_factor_neg = 1 - (0.91 * iteration / 400000.0)

    contrast_factor_pos = 1 + (0.5*iteration/350000.0)
    contrast_factor_neg = 1 - (0.5 * iteration / 400000.0)


    #print ('iteration',iteration,'Augment Status ',frequency_factor,color_factor,dropout_factor,blur_factor,add_factor,
    #    multiply_factor_pos,multiply_factor_neg,contrast_factor_pos,contrast_factor_neg)


    augmenter = iaa.Sequential([

        iaa.Sometimes(frequency_factor, iaa.GaussianBlur((0, blur_factor))),
        # blur images with a sigma between 0 and 1.5
        iaa.Sometimes(frequency_factor, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0,dropout_factor ),
                                                                  per_channel=color_factor)),
        # add gaussian noise to images
        iaa.Sometimes(frequency_factor, iaa.CoarseDropout((0.0, dropout_factor), size_percent=(
            0.08, 0.2), per_channel=color_factor)),
        # randomly remove up to X% of the pixels
        iaa.Sometimes(frequency_factor, iaa.Dropout((0.0, dropout_factor), per_channel=color_factor)),
        # randomly remove up to X% of the pixels
        iaa.Sometimes(frequency_factor,
                      iaa.Add((-add_factor, add_factor), per_channel=color_factor)),
        # change brightness of images (by -X to Y of original value)
        iaa.Sometimes(frequency_factor,
                      iaa.Multiply((multiply_factor_neg, multiply_factor_pos), per_channel=color_factor)),
        # change brightness of images (X-Y% of original value)
        iaa.Sometimes(frequency_factor, iaa.ContrastNormalization((contrast_factor_neg, contrast_factor_pos),
                                                                       per_channel=color_factor)),
        # improve or worsen the contrast
        iaa.Sometimes(frequency_factor, iaa.Grayscale((0.0, 1))),  # put grayscale

    ],
        random_order=True  # do all of the above in random order
    )

    return augmenter

def medium_harder(image_iteration):


    iteration = image_iteration / 120
    frequency_factor = 0.05 + float(iteration)/1000000.0
    color_factor = float(iteration)/1000000.0
    dropout_factor = 0.198667 + (0.03856658 - 0.198667) / (1 + (iteration / 196416.6) ** 1.863486)

    blur_factor = 0.5 + (0.5*iteration/100000.0)

    add_factor = 10 + 10*iteration/150000.0

    multiply_factor_pos = 1 + (2.5*iteration/500000.0)
    multiply_factor_neg = 1 - (0.91 * iteration / 500000.0)

    contrast_factor_pos = 1 + (0.5*iteration/500000.0)
    contrast_factor_neg = 1 - (0.5 * iteration / 500000.0)


    #print 'Augment Status ',frequency_factor,color_factor,dropout_factor,blur_factor,add_factor,\
    #    multiply_factor_pos,multiply_factor_neg,contrast_factor_pos,contrast_factor_neg


    augmenter = iaa.Sequential([

        iaa.Sometimes(frequency_factor, iaa.GaussianBlur((0, blur_factor))),
        # blur images with a sigma between 0 and 1.5
        iaa.Sometimes(frequency_factor, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0,dropout_factor ),
                                                                  per_channel=color_factor)),
        # add gaussian noise to images
        iaa.Sometimes(frequency_factor, iaa.CoarseDropout((0.0, dropout_factor), size_percent=(
            0.08, 0.2), per_channel=color_factor)),
        # randomly remove up to X% of the pixels
        iaa.Sometimes(frequency_factor, iaa.Dropout((0.0, dropout_factor), per_channel=color_factor)),
        # randomly remove up to X% of the pixels
        iaa.Sometimes(frequency_factor,
                      iaa.Add((-add_factor, add_factor), per_channel=color_factor)),
        # change brightness of images (by -X to Y of original value)
        iaa.Sometimes(frequency_factor,
                      iaa.Multiply((multiply_factor_neg, multiply_factor_pos), per_channel=color_factor)),
        # change brightness of images (X-Y% of original value)
        iaa.Sometimes(frequency_factor, iaa.ContrastNormalization((contrast_factor_neg, contrast_factor_pos),
                                                                       per_channel=color_factor)),
        # improve or worsen the contrast
        iaa.Sometimes(frequency_factor, iaa.Grayscale((0.0, 1))),  # put grayscale

    ],
        random_order=True  # do all of the above in random order
    )

    return augmenter

def hard_harder(image_iteration):


    iteration = image_iteration / 120
    frequency_factor = min(0.05 + float(iteration)/200000.0, 1.0)
    color_factor = float(iteration)/1000000.0
    dropout_factor = 0.198667 + (0.03856658 - 0.198667) / (1 + (iteration / 196416.6) ** 1.863486)

    blur_factor = 0.5 + (0.5*iteration/100000.0)

    add_factor = 10 + 10*iteration/100000.0

    multiply_factor_pos = 1 + (2.5*iteration/200000.0)
    multiply_factor_neg = 1 - (0.91 * iteration / 500000.0)

    contrast_factor_pos = 1 + (0.5*iteration/500000.0)
    contrast_factor_neg = 1 - (0.5 * iteration / 500000.0)


    #print 'Augment Status ',frequency_factor,color_factor,dropout_factor,blur_factor,add_factor,\
    #    multiply_factor_pos,multiply_factor_neg,contrast_factor_pos,contrast_factor_neg


    augmenter = iaa.Sequential([

        iaa.Sometimes(frequency_factor, iaa.GaussianBlur((0, blur_factor))),
        # blur images with a sigma between 0 and 1.5
        iaa.Sometimes(frequency_factor, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0,dropout_factor ),
                                                                  per_channel=color_factor)),
        # add gaussian noise to images
        iaa.Sometimes(frequency_factor, iaa.CoarseDropout((0.0, dropout_factor), size_percent=(
            0.08, 0.2), per_channel=color_factor)),
        # randomly remove up to X% of the pixels
        iaa.Sometimes(frequency_factor, iaa.Dropout((0.0, dropout_factor), per_channel=color_factor)),
        # randomly remove up to X% of the pixels
        iaa.Sometimes(frequency_factor,
                      iaa.Add((-add_factor, add_factor), per_channel=color_factor)),
        # change brightness of images (by -X to Y of original value)
        iaa.Sometimes(frequency_factor,
                      iaa.Multiply((multiply_factor_neg, multiply_factor_pos), per_channel=color_factor)),
        # change brightness of images (X-Y% of original value)
        iaa.Sometimes(frequency_factor, iaa.ContrastNormalization((contrast_factor_neg, contrast_factor_pos),
                                                                       per_channel=color_factor)),
        # improve or worsen the contrast
        iaa.Sometimes(frequency_factor, iaa.Grayscale((0.0, 1))),  # put grayscale

    ],
        random_order=True  # do all of the above in random order
    )

    return augmenter


def soft_harder(image_iteration):

    iteration = image_iteration / 120
    frequency_factor = 0.05 + float(iteration)/1200000.0
    color_factor = float(iteration)/1200000.0
    dropout_factor = 0.198667 + (0.03856658 - 0.198667) / (1 + (iteration / 196416.6) ** 1.863486)

    blur_factor = 0.5 + (0.5*iteration/120000.0)

    add_factor = 10 + 10*iteration/170000.0

    multiply_factor_pos = 1 + (2.5*iteration/800000.0)
    multiply_factor_neg = 1 - (0.91 * iteration / 800000.0)

    contrast_factor_pos = 1 + (0.5*iteration/800000.0)
    contrast_factor_neg = 1 - (0.5 * iteration / 800000.0)

    augmenter = iaa.Sequential([

        iaa.Sometimes(frequency_factor, iaa.GaussianBlur((0, blur_factor))),
        # blur images with a sigma between 0 and 1.5
        iaa.Sometimes(frequency_factor, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0,dropout_factor ),
                                                                  per_channel=color_factor)),
        # add gaussian noise to images
        iaa.Sometimes(frequency_factor, iaa.CoarseDropout((0.0, dropout_factor), size_percent=(
            0.08, 0.2), per_channel=color_factor)),
        # randomly remove up to X% of the pixels
        iaa.Sometimes(frequency_factor, iaa.Dropout((0.0, dropout_factor), per_channel=color_factor)),
        # randomly remove up to X% of the pixels
        iaa.Sometimes(frequency_factor,
                      iaa.Add((-add_factor, add_factor), per_channel=color_factor)),
        # change brightness of images (by -X to Y of original value)
        iaa.Sometimes(frequency_factor,
                      iaa.Multiply((multiply_factor_neg, multiply_factor_pos), per_channel=color_factor)),
        # change brightness of images (X-Y% of original value)
        iaa.Sometimes(frequency_factor, iaa.ContrastNormalization((contrast_factor_neg, contrast_factor_pos),
                                                                       per_channel=color_factor)),
        # improve or worsen the contrast
        iaa.Sometimes(frequency_factor, iaa.Grayscale((0.0, 1))),  # put grayscale

    ],
        random_order=True  # do all of the above in random order
    )

    return augmenter


class ColorRandomization(object):

    def __init__(self, iteration):
        super(ColorRandomization, self).__init__()
        self.id = iteration % 4 # hard coded for now
        self.augmentation_allowed = ['red', 'green', 'blue', 'none']
        self.color_code = {'red': (255, 0, 0),
                           'green': (0, 255, 0),
                           'blue': (0, 0, 255),
                           'none': (0, 0, 0)
                           }

    def augment_image(self, image, **kwargs):
        segment = kwargs.get('seg_map')
        segment[:,:,0] = segment[:,:,2]
        segment[:,:,1] = segment[:,:,2]
        mask = segment==1
        new_image = image*mask
        aug = self.augmentation_allowed[self.id]
        if aug == 'none':
            mask = 0
        r, g, b = self.color_code[aug]
        new_image[:,:,0] = r
        new_image[:,:,1] = g
        new_image[:,:,2] = b
        image = image*(1-mask) + new_image*mask
        return image


class TextureRandomization(object):

    def __init__(self, iteration):
        super(TextureRandomization, self).__init__()
        self.id = iteration%4 # hardcoded to include [0, 25, 50, 75, 100]% actual samples

    def augment_image(self, image, **kwargs):
        segment = kwargs.get('seg_map')
        texture = kwargs.get('texture')
        segment[:,:,0] = segment[:,:,2]
        segment[:,:,1] = segment[:,:,2]
        mask = ((segment==1)+(segment==9)+(segment==0)).astype(bool)
        mask *= (self.id>2)
        image = image*(1-mask) + texture*mask
        return image


class TextureAlphaRandomization(object):

    def __init__(self, iteration):
        super(TextureAlphaRandomization, self).__init__()
        self.id = iteration%4 # hardcoded to include [0, 25, 50, 75, 100]% actual samples
        self.alpha = 0.50

    def augment_image(self, image, **kwargs):
        segment = kwargs.get('seg_map')
        texture = kwargs.get('texture')
        segment[:,:,0] = segment[:,:,2]
        segment[:,:,1] = segment[:,:,2]
        mask = segment==1
        mask *= (self.id>2)
        alpha_image = (image*(1-self.alpha) + texture*self.alpha).astype(int)
        image = (image*(1-mask) + alpha_image*mask).astype(int)
        return image


class TexturePatchRandomization(object):
    # harcoded for patches of size (22,25)
    def __init__(self, iteration):
        super(TexturePatchRandomization, self).__init__()
        self.id = iteration%4 # hardcoded to include [0, 25, 50, 75, 100]% actual samples

    def augment_image(self, image, **kwargs):
        segment = kwargs.get('seg_map')
        texture = kwargs.get('texture')
        mask = (segment==7).astype(int)
        mask = mask[:,:,2]
        h, w = mask.shape
        mask = mask.reshape(h//22,22,w//25,25).swapaxes(1,2).reshape(-1,22,25)
        mask_sum = mask.reshape(-1, 22*25)
        mask_sum = mask_sum.sum(1)
        mask_sum = (mask_sum==22*25).astype(int)
        relevant_mask = np.append(np.where(mask_sum==1)[0], [0], 0)
        mask_index = np.random.choice(relevant_mask, 1, replace=False)
        mask_sum = np.zeros(len(mask_sum))
        mask_sum[mask_index] = 1
        mask_extended = np.stack([mask_sum]*22*25,1).reshape(-1,22,25)
        mask_new = mask*mask_extended
        mask_new = mask_new.reshape(h//22,w//25,22,25).swapaxes(1,2).reshape(h,w)
        mask_new = np.stack([mask_new]*3, 2)
        mask_new *= (self.id>2)
        image = (image*(1-mask_new)+texture*mask_new).astype(int)
        return image