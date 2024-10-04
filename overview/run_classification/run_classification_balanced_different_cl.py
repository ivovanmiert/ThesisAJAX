import sys 
sys.path.insert(0, '/home/ivanmiert/NEW_OBJECT/CLASSIFICATION_MODEL')
import argparse

"""
This file was used to run the training/validation/testing of the classification model 
One of two functions were imported (with the same name): train_model, evaluate_model
Based on the task one of the two was imported. 
Regarding the one from: example5_overview_balanced_undersampling_01_more_layers_less_features_added_all_5. The functions imported were solely used to load all training/valdiation/test data for the first time. After this, no training was done, just the loading. 
This one-time loading was done since the loading of the concatenated features took about 5 minutes for 79 features, and 8 minutes for 130 features. This way of saving the data in the correct format (with the labels), it only took 1 minute of reloading. 
This reloading was done before the training/validation/testing as done in the imported functions from: example5_overview_balanced_loaded_different_cl2

"""
#The right import of the function would be made based on which function to use from which file. The above one was used when the datasets were loaded for the first time. 
#The second import was used during the training of the models, in which the already loaded data was reloaded
#The files for this functions are in the CLASSIFICATION_MODEL folder/class as is shown in the path import above. 

#Import one:
#from example5_overview_balanced_undersampling_01_more_layers_less_features_added_all_5 import train_model, evaluate_model

#Import two:
from example5_overview_balanced_loaded_different_cl2 import train_model, evaluate_model
import yaml
import torch
import os


def load_config(config_path='/home/ivanmiert/overview/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for classification model")
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of LSTM layers.')
    parser.add_argument('--num_layers', type=int, default=100, help='Number of LSTM layers.')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for the LSTM model.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the optimizer.')
    parser.add_argument('--clip_value', type=float, default=5, help='Gradient clipping value.')
    parser.add_argument('--classification_sort', type=str, default='primary')
    parser.add_argument('--hpe_sort', type=str, default='basic')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config() 


    print(f"hpe_sort {args.hpe_sort}")
    labels = config['output_files']['labels_multiple_file']
    num_layers = config['classification']['number_of_layers']
    num_features = config['classification']['number_of_features']
    classification_sort = args.classification_sort
    model_save_path_template = config['classification']['model_save_template']
    dataframe_save_path_template = config['classification']['dataframe_save_path']

    if classification_sort == 'primary':
        print('hallo_2')
        model_save_path = model_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        dataframe_save_path = dataframe_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        relevant_column = 'label'
        relevant_labels = [0, 1, 2, 3, 4]

    if classification_sort == 'shot_body_part':
        model_save_path = model_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        dataframe_save_path = dataframe_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        relevant_column = 'body_part_label'
        relevant_labels = [1, 2, 3]

    if classification_sort == 'duel':
        model_save_path = model_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        dataframe_save_path = dataframe_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        relevant_column = 'duel_label'
        relevant_labels = [1, 2, 3]

    if classification_sort == 'normal_pass_or_cross':
        print('goed')
        model_save_path = model_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        dataframe_save_path = dataframe_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        relevant_column = 'pass_label'
        relevant_labels = [1, 2]

    if classification_sort == 'accuracy_cross':
        model_save_path = model_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        dataframe_save_path = dataframe_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        relevant_column = 'cross_accurate'
        relevant_labels = [1, 2]

    if classification_sort == 'cross_direction':
        model_save_path = model_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        dataframe_save_path = dataframe_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        relevant_column = 'cross_direction'
        relevant_labels = [1, 2, 3]

    if classification_sort == 'cross_flank':
        model_save_path = model_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        dataframe_save_path = dataframe_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        relevant_column = 'cross_flank'
        relevant_labels = [1, 2, 3]

    if classification_sort == 'pass_accurate':
        model_save_path = model_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        dataframe_save_path = dataframe_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        relevant_column = 'pass_accurate'
        relevant_labels = [1, 2]

    if classification_sort == 'pass_direction':
        model_save_path = model_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        dataframe_save_path = dataframe_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        relevant_column = 'pass_direction'
        relevant_labels = [1, 2, 3]

    if classification_sort == 'pass_distance':
        model_save_path = model_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        dataframe_save_path = dataframe_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        relevant_column = 'pass_distance'
        relevant_labels = [1, 2]

    if classification_sort == 'pass_progressive':
        model_save_path = model_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        dataframe_save_path = dataframe_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        relevant_column = 'pass_progressive'
        relevant_labels = [1, 2]

    if classification_sort == 'pass_through':
        model_save_path = model_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        dataframe_save_path = dataframe_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        relevant_column = 'pass_through'
        relevant_labels = [1, 2]

    if classification_sort == 'shot_on_target':
        model_save_path = model_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        dataframe_save_path = dataframe_save_path_template.format(args.hpe_sort, classification_sort, num_features, args.hidden_size, args.batch_size, args.learning_rate, args.dropout_rate, args.num_layers)
        relevant_column = 'shot_on_target'
        relevant_labels = [1, 2]

    if args.hpe_sort == 'top_down':
        model, test_dataset = train_model(
            '/scratch-shared/ivanmiert/overview/train_testit_top_down_2.pt',
            '/scratch-shared/ivanmiert/overview/validation_testit_top_down_2.pt',
            '/scratch-shared/ivanmiert/overview/test_set_testit_top_down_2.pt', 
            '/scratch-shared/ivanmiert/overview/original_dataset_testit_top_down_2.pt',             
            model_save_path=model_save_path,  
            num_layers=args.num_layers, 
            num_features=130, 
            relevant_column=relevant_column, 
            relevant_labels=relevant_labels, 
            hidden_size=args.hidden_size, 
            batch_size=args.batch_size, 
            epochs=args.epochs, 
            learning_rate=args.learning_rate, 
            weight_decay=args.weight_decay, 
            clip_value=args.clip_value,
            dropout_rate = args.dropout_rate,
            classification_sort = args.classification_sort
        )


    elif args.hpe_sort == 'bottom_up':
        model, test_dataset = train_model(
            '/scratch-shared/ivanmiert/overview/train_testit_bottom_up_2.pt',
            '/scratch-shared/ivanmiert/overview/validation_testit_bottom_up_2.pt',
            '/scratch-shared/ivanmiert/overview/test_set_testit_bottom_up_2.pt', 
            '/scratch-shared/ivanmiert/overview/original_dataset_testit_bottom_up_2.pt',             
            model_save_path=model_save_path, 
            num_layers=args.num_layers, 
            num_features=130, 
            relevant_column=relevant_column, 
            relevant_labels=relevant_labels, 
            hidden_size=args.hidden_size, 
            batch_size=args.batch_size, 
            epochs=args.epochs, 
            learning_rate=args.learning_rate, 
            weight_decay=args.weight_decay, 
            clip_value=args.clip_value,
            dropout_rate = args.dropout_rate, 
            classification_sort = args.classification_sort
        )
            
    elif args.hpe_sort == 'basic':
        model, test_dataset = train_model(
            train_path='/scratch-shared/ivanmiert/overview/train_testit_low_2.pt',
            val_path='/scratch-shared/ivanmiert/overview/validation_low_2.pt',
            test_path='/scratch-shared/ivanmiert/overview/test_set_low_2.pt', 
            original_dataset_path='/scratch-shared/ivanmiert/overview/original_dataset_low_2.pt', 
            # train_path = '/scratch-shared/ivanmiert/overview/train_testit.pt',
            # val_path = '/scratch-shared/ivanmiert/overview/validation_testit.pt',
            # test_path = '/scratch-shared/ivanmiert/overview/test_set_testit.pt',
            # original_dataset_path = '/scratch-shared/ivanmiert/overview/original_dataset_testit.pt',

            model_save_path=model_save_path,  # Ensure model_save_path is properly formatted above
            num_layers=args.num_layers, 
            num_features=79, 
            relevant_column=relevant_column, 
            relevant_labels=relevant_labels, 
            hidden_size=args.hidden_size, 
            batch_size=args.batch_size, 
            epochs=args.epochs, 
            learning_rate=args.learning_rate, 
            weight_decay=args.weight_decay, 
            clip_value=args.clip_value,
            dropout_rate = args.dropout_rate,
            classification_sort = args.classification_sort
        )

    #model, test_dataset = train_model(concatenated_features, labels, epochs=10)
    number_of_classes = len(relevant_labels)
    evaluate_model(model, test_dataset, number_of_classes, dataframe_save_path, relevant_column, hidden_size=args.hidden_size, batch_size_train=args.batch_size, learning_rate=args.learning_rate, dropout_rate=args.dropout_rate, number_of_layers=args.num_layers, classification_sort=classification_sort)
    save_path_model = f"/scratch-shared/ivanmiert/overview/model_{args.hidden_size}_{args.batch_size}_{args.learning_rate}_{args.dropout_rate}_{args.num_layers}.pt"
    torch.save(model.state_dict(), save_path_model)
    print(f"model saved to: {save_path_model}")

if __name__ == "__main__":
    main()