from cleanfid import fid

real_data_root = "/data/jiahuic/ImageNetLT_val_test/"
synth_data_root = "/datastor1/jiahuikchen/synth_ImageNet/"

# first in tup is always name of real data folder
comparison_dirs = [
    # 30 subset gets numerical error when calculating with ImageNet_LT_test_30
    ("ImageNet_LT_test", "cutmix_30"),
    ("ImageNet_LT_test", "dropout_30"),
    ("ImageNet_LT_test", "embed_cutmix_30"),
    ("ImageNet_LT_test", "embed_mixup_30"),
    ("ImageNet_LT_test", "mixup_30"),
    ("ImageNet_LT_test", "rand_img_cond_30"), 
    # ("ImageNet_LT_test_90", "cutmix_90"),
    # ("ImageNet_LT_test_90", "dropout_90"),
    # ("ImageNet_LT_test_90", "embed_cutmix_90"),
    # ("ImageNet_LT_test_90", "embed_mixup_90"),
    # ("ImageNet_LT_test_90", "mixup_90"),
    # ("ImageNet_LT_test_90", "rand_img_cond_90"),
]

for comparison in comparison_dirs:
    real_data_folder = comparison[0]
    synth_data_folder = comparison[1]
    score = fid.compute_fid(f"{real_data_root}{real_data_folder}", f"{synth_data_root}{synth_data_folder}")
    print(f"{synth_data_folder} and {real_data_folder} FID: \t {score}")