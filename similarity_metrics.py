import numpy as np
from numpy import resize
import pandas as pd
import os
from os import listdir
from skimage import io
import glob
from xlsx_file import get_excel_file
from sewar.full_ref import mse, rmse, psnr, ssim, uqi, msssim, vifp, rmse_sw
import psnrhmam
import argparse
from statistic_test import statistic_test
import sys
import constants
import cv2
import matplotlib.pyplot as plt

def obtain_similarity_metrics(GT_img, distorted_img):
    # MEAN SQUARED ERROR
    mse_value = mse(GT_img, distorted_img)
    # STRUCTURAL SIMILARITY
    ssim_value = ssim(GT_img, distorted_img)
    # PEAK SIGNAL TO NOISE RATIO
    psnr_value = psnr(GT_img, distorted_img)
    # ROOT MEAN SQUARED ERROR
    rmse_value = rmse(GT_img, distorted_img)
    # VISUAL INFORMATION FIDELITY
    vif_value = vifp(GT_img, distorted_img)
    # UNIVERSAL IMAGE QUALITY INDEX
    uqi_value = uqi(GT_img, distorted_img)
    # MULTI-SCALE STRUCTURAL SIMILARITY INDEX
    msssim_value = msssim(GT_img, distorted_img)
    # PSNR-HVS-M  &  PSNR-HVS
    p_hvs_m, p_hvs = psnrhmam.color_psnrhma(GT_img, distorted_img)

    return mse_value, ssim_value, psnr_value, rmse_value, vif_value, uqi_value, msssim_value, p_hvs_m, p_hvs


def main():
    """
    execution example:
    - python similarity_metrics.py --path_run "images/" --metrics_excel_name 'metrics.xlsx' --test_excel_name 'statistic_test.xlsx' --methods_used 6
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', default="images/", help='path to the run folder.')
    parser.add_argument('--metrics_excel_name', default="metrics.xlsx", help='metrics excel name + .xlsx')
    parser.add_argument('--test_excel_name', default="statistic_test.xlsx", help='statistic test excel name + .xlsx')
    parser.add_argument('--methods_used', default=6, type=int,  help='different methods used')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run
    metrics_excel_name = parsed_args.metrics_excel_name
    test_excel_name = parsed_args.test_excel_name
    methods_used = parsed_args.methods_used

    workbook, worksheet = get_excel_file(metrics_excel_name, constants.sheet_name)

    folders_dir = 1

    for case_folder in listdir(path_run):
        modified_idx = 0

        original_img = io.imread(os.path.join(path_run, case_folder, constants.reference_img))
        x_size = original_img.shape[0]
        y_size = original_img.shape[1]

        for impainted_img in (listdir(os.path.join(path_run, case_folder))):
            if impainted_img.startswith(constants.distorted_img):
                modified_idx += 1

                modified_img = cv2.resize(io.imread(os.path.join(path_run, case_folder, impainted_img)), (y_size, x_size))

                mse_value, ssim_value, psnr_value, rmse_value, vif_value, uqi_value, msssim_value, p_hvs_m_value, p_hvs_value = obtain_similarity_metrics(original_img, modified_img)

                metrics = (
                    ['MSE', mse_value],
                    ['SSIM', np.mean(ssim_value[:])],
                    ['PSNR', psnr_value],
                    ['RMSE', rmse_value],
                    ['VIF', vif_value],
                    ['UQI', uqi_value],
                    ['MSSSIM', msssim_value.real],
                    ['PSNR-HVS-M', p_hvs_m_value],
                    ['PSNR-HVS', p_hvs_value],
                )

                for metric in range(len(metrics)):
                    worksheet.cell(row=1, column=metric + 3).value = metrics[metric][0]
                    worksheet.cell(row=folders_dir+modified_idx+1, column=metric + 3).value = metrics[metric][1]

                worksheet.cell(row=folders_dir+modified_idx+1, column=2).value = os.path.splitext(impainted_img)[0]

        worksheet.cell(row=folders_dir+1, column=1).value = os.path.split(case_folder)[1]
        folders_dir += methods_used + 1

    workbook.save(metrics_excel_name)

    dfs = pd.read_excel(metrics_excel_name, sheet_name=constants.sheet_name)

    for metric in range(2, dfs.shape[1]):
        statistic_test(dfs, dfs.columns[metric], test_excel_name, methods_used)


if __name__ == "__main__":
    main()
