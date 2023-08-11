![image](https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo_wide.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

## RFM Segmentation
Not every customer has the same revenue potential for a given retail organization.  By acknowledging this fact, retailers can better tailer their engagement to ensure the profitability of their relationship with a customer.  But how do we distinguish between higher and lower value customers? And now might we identify specific behaviors to address in order to turn good customers into great ones?

Today, we often address this concern with an estimation of customer lifetime value (CLV).  But while CLV estimations can be incredibly helpful, we often don't need precise revenue estimates when deciding which customers to engage with which offers.  Instead, a lightweight approach that examines the recency, frequency and (per-interaction) monetary value of a given customer can go a long way to divide customers into groups of higher, lower and in-between value, and this is exactly what the practice of [RFM segmentation](https://link.springer.com/content/pdf/10.1057/palgrave.jdm.3240019.pdf) provides.

Pre-dating the formal CLV techniques frequently used today, RFM segmentation remains surprisingly popular with marketing teams looking to quickly organize customers into value-aligned groupings.  In this notebook, we want to demonstrate how an RFM segmentation can be performed and operationalized to enable personalized workflows.
___
<bryan.smith@databricks.com>

___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| openpyxl | Python library to read/write Excel 2010 xlsx/xlsm/xltx/xltm files| MIT | https://pypi.org/project/openpyxl/ |

## Getting started

Although specific solutions can be downloaded as .dbc archives from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries. 

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

To start using a solution accelerator in Databricks simply follow these steps: 

1. Clone solution accelerator repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
2. Attach the `RUNME` notebook to any cluster and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. The job configuration is written in the RUNME notebook in json format. 
3. Execute the multi-step-job to see how the pipeline runs. 
4. You might want to modify the samples in the solution accelerator to your need, collaborate with other users and run the code samples against your own data. To do so start by changing the Git remote of your repository  to your organization’s repository vs using our samples repository (learn more). You can now commit and push code, collaborate with other user’s via Git and follow your organization’s processes for code development.

The cost associated with running the accelerator is the user's responsibility.


## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
