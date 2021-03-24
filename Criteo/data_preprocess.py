import category_encoders as ce

def prepare_categorical_inputs_target_encoder(categorical_train_df, categorical_test_df, categorical_feature_names, train_y):
    encoder_purpose = ce.TargetEncoder(cols=categorical_feature_names, smoothing=10)
    encoder_df = encoder_purpose.fit_transform(categorical_train_df[categorical_feature_names], train_y)
    print("encoder_train_df.shape: {}".format(encoder_df.shape))
    print("encoder_train_df head 10: {}".format(encoder_df.head(10)))
    test_df = encoder_purpose.transform(categorical_test_df[categorical_feature_names])
    print("encoder_test_df.shape: {}".format(test_df.shape))
    print("encoder_test_df head 10: {}".format(test_df.head(10)))
    return (encoder_df.values, test_df.values)

def prepare_categorical_inputs(categorical_df, categrorial_feature_names, y=None):
    # encoder_purpose = ce.HashingEncoder(n_components=32, cols=categrorial_feature_names)
    encoder_purpose = ce.TargetEncoder(cols=categrorial_feature_names, smoothing=10)
    encoder_df = encoder_purpose.fit_transform(categorical_df[categrorial_feature_names], y)
    print("encoder_df.shape: {}".format(encoder_df.shape))
    print("encoder_df head 10: {}".format(encoder_df.head(10)))
    return encoder_df.values

def prepare_categorical_inputs_hashing_encoder(categorical_train_df, categorical_test_df, categorical_feature_names, n_components=128):
    encoder_purpose = ce.HashingEncoder(cols=categorical_feature_names, n_components=n_components)
    encoder_df = encoder_purpose.fit_transform(categorical_train_df[categorical_feature_names])
    print("encoder_train_df.shape: {}".format(encoder_df.shape))
    print("encoder_train_df head 10: {}".format(encoder_df.head(10)))
    test_df = encoder_purpose.transform(categorical_test_df[categorical_feature_names])
    print("encoder_test_df.shape: {}".format(test_df.shape))
    print("encoder_test_df head 10: {}".format(test_df.head(10)))
    return (encoder_df.values, test_df.values)
