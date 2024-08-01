# kan-intro
Kolmogorov-Arnold Network (KAN) introduction

## Install

```
pip install -r requirements.txt
```

## TODO

Kopiere service_request_kan_train.ipynb til test kun for en enkelt person, Nils eller Ola
Finne ut hvorfor jeg får feilmelding
MultKAN.py:409: UserWarning: std(): degrees of freedom is <= 0. Correction should be strictly less than the reduction factor (input numel divided by output numel). (Triggered internally at ../aten/src/ATen/native/ReduceOps.cpp:1760.)
output_range = torch.std(postacts, dim=0) # for visualization, include the contribution from both spline + symbolic
