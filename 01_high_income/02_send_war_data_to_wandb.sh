#!/bin/bash

PROJECT_NAME="high_income"
FILE_NAME="raw_data.csv"
ARTIFACT_TYPE="data"
ARTIFACT_DESC="The raw_data from 1994 US Census"

wandb login --relogin
wandb artifact put --name $PROJECT_NAME"/"$FILE_NAME \
                   --type $ARTIFACT_TYPE \
                   --description "${ARTIFACT_DESC}" "build/"$FILE_NAME
