azcopy_linux_amd64_10.14.1/azcopy copy https://biglmdiag.blob.core.windows.net/vinvl/pretrain_corpus/coco_flickr30k_gqa.tsv .

azcopy_linux_amd64_10.14.1/azcopy copy https://biglmdiag.blob.core.windows.net/vinvl/pretrain_corpus/coco_flickr30k_gqa.lineidx .

# image labels
azcopy_linux_amd64_10.14.1/azcopy copy https://biglmdiag.blob.core.windows.net/vinvl/pretrain_corpus/X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2/ . --recursive

# image features
azcopy_linux_amd64_10.14.1/azcopy copy https://biglmdiag.blob.core.windows.net/vinvl/image_features/coco_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2 image_features/ --recursive

azcopy_linux_amd64_10.14.1/azcopy copy https://biglmdiag.blob.core.windows.net/vinvl/image_features/flickr30k_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2 image_features/ --recursive

azcopy_linux_amd64_10.14.1/azcopy copy https://biglmdiag.blob.core.windows.net/vinvl/image_features/gqa_X152C4_frcnnbig2_exp168model_0060000model.roi_heads.nm_filter_2_model.roi_heads.score_thresh_0.2 image_features/ --recursive
