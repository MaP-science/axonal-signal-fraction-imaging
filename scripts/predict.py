import sys
from asfimaging.pipeline.ASFpipeline import ASFpipeline



def predict(config_name, use_features=False):
    # Run the ASF pipeline for calculating features, predict the maps and save maps
    ASFImaging = ASFpipeline(config_name)
    raw_data, raw_affine, mask = ASFImaging.data_extractor()
    if not use_features:
        features = ASFImaging.real_feature_extractor(raw_data, raw_affine, mask)
        labels = ASFImaging.predict_from_features(features)
    else:
        labels = ASFImaging.get_premade_features()
    dpar, dperp, asf_cal, asf_pre = ASFImaging.save_maps(labels)
    return dpar, dperp, asf_cal, asf_pre

if __name__ == "__main__":
    config_name = sys.argv[1]
    use_pre_features = False
    if len(sys.argv) > 2:
        use_pre_features = sys.argv[2].lower() == True
        
    dpar, dperp, asf_cal, asf_pre = predict(config_name, use_pre_features)

    print("Create maps for the DWI data")

