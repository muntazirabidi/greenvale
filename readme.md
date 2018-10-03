# Greenvale midsize band classification KNN model
This project is the work undertaken by Catalyst AI for Greenvale involving the classification of potatoes into midsize bands. This was achieved by using only the weight of the sample to be classified and using a KNN classifier fed historical data to decide the most likely band the sample should fall in.

## Getting Started

### Prerequisities


In order to run this container you'll need docker installed.

* [Windows](https://docs.docker.com/windows/started)
* [OS X](https://docs.docker.com/mac/started/)
* [Linux](https://docs.docker.com/linux/started/)

### Usage

Simply build the container

```shell
docker build -t greenvale-knn .
```

And run it whilst opening port 8000

```shell
docker run -p 8000:8000 greenvale-knn
```

## API
There are currently two api endpoints offering a simple response with additional statistics and another that includes mu, CoV, and k in the response.

### Simple Classify
Takes an array of tuber samples and returns their classified midsize bands.

* **URL**  
`/classify`
* **Method**  
`POST`
* **Data Params**  
Currently the classifier uses the first sample as reference to the variety being classified. This creates the requirement that for any single request, all the samples must be of the same variety.
  ```
  [ 
    { 
      "variety": "MarisPiper", 
      "tuber_id": "00001", 
      "sample_id": "510c1fb0", 
      "tuber_weight": 113.1 
    }, 
    { 
      "variety": "MarisPiper", 
      "tuber_id": "00002", 
      "sample_id": "510c1fb0", 
      "tuber_weight": 103.3 
    } 
  ]
  ```
* **Response**  
  * **Code:** `200`  
  **Content**
      ```
      [ 
        { 
            "sample_id": "510c1fb0", 
            "tuber_details": [ 
                { 
                    "variety": "MarisPiper", 
                    "tuber_weight": 20.1, 
                    "size_band": 27.5, 
                    "tuber_id": "00154" 
                } 
            ] 
        } 
      ]
      ```

### Expanded Classify
Takes an array of tuber samples and returns their classified midsize bands, along with calculated mu, CoV, and k for the input sample collection.

* **URL**  
`/expanded-classify`
* **Method**  
`POST`
* **Data Params**  
Currently the classifier uses the first sample as reference to the variety being classified. This creates the requirement that for any single request, all the samples must be of the same variety.
  ```
  [ 
    { 
      "variety": "MarisPiper", 
      "tuber_id": "00001", 
      "sample_id": "510c1fb0", 
      "tuber_weight": 113.1 
    }, 
    { 
      "variety": "MarisPiper", 
      "tuber_id": "00002", 
      "sample_id": "510c1fb0", 
      "tuber_weight": 103.3 
    } 
  ]
  ```
* **Response**  
  * **Code:** `200`  
  **Content**
      ```
      [ 
        "statistics": { 
          "k": 117.85346667372748, 
          "mu": 39.62871287128713, 
          "CoV": 16.68818509636155 
        },
        "samples": [ 
          { 
            "sample_id": "510c1fb0", 
            "tuber_details": [ 
                { 
                    "variety": "MarisPiper", 
                    "tuber_weight": 20.1, 
                    "size_band": 27.5, 
                    "tuber_id": "00154" 
                } 
            ] 
          } 
        ] 
      ]
      ```
## Updating the training data
The training data is stored per variety in `.csv` files within the `training_data` folder. If more data is avaiable it can be added to the end of the csv in any order as long as the format is maintained.

When adding new varieties, the variety name and k-value for the knn classifier should be added to the `constants.py` file. Also, the data relevant to the variety should be added to a new file named `{variety_name}_TuberData.csv` with the data it contains in the same format as the pre-existing files.
