from collections import OrderedDict

label2text_dict_abdomenct = OrderedDict([
    # [0,  'background'], # 0: background added later after text description
    [1,  'spleen'],
    [2,  'right kidney'],
    [3,  'left kidney'],
    [4,  'gall bladder'],
    [5,  'esophagus'],
    [6,  'liver'],
    [7,  'stomach'],
    [8,  'aorta'],
    [9,  'inferior vena cava'],
    [10, 'portal and splenic vein'],
    [11, 'pancreas'],
    [12, 'left adrenal gland'],
    [13, 'right adrenal gland'],
])


label2text_dict_cardiac = OrderedDict([
    # [0,  'background'], # 0: background added later after text description
    [1,  'right ventricle'],
    [2,  'left ventricle myocardium'],
    [3,  'left ventricle blood pool'],
])