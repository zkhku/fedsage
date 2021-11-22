# To run the code
1. Unzip the package to your local directory;
2. Run 'pip install -r requirements.txt' to download required packages;
3. Open file ~/nips_code/src/utils/config.py;
4. Replace the "change_to_your_current_path" in line 2 of config.py
(**root_path= "change_to_your_current_path"**) to your current path;
    - You can change hyper-parameters in config.py according to different testing scenarios;
5. Run the whole pipline with 'python ~/nips_code/src/system_test.py'.


# If you find this work useful for your research, please cite
```
@inproceedings{zhang2021subgraph,
  title={Subgraph federated learning with missing neighbor generation},
  author={Zhang, Ke and Yang, Carl and Li, Xiaoxiao and Sun, Lichao and Yiu, Siu Ming},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```
