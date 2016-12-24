'''
 * Copyright (C) 2016  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of jingjuPhoneticSegmentationHMM
 *
 * pypYIN is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 *
 * If you have any problem about this python version code, please contact: Rong Gong
 * rong.gong@upf.edu
 *
 *
 * If you want to refer this code, please use this article:
 *
'''

import os


def getRecordings(wav_path):
    recordings      = []
    for root, subFolders, files in os.walk(wav_path):
            for f in files:
                file_prefix, file_extension = os.path.splitext(f)
                if file_prefix != '.DS_Store' and file_prefix != '_DS_Store':
                    recordings.append(file_prefix)

    return recordings

def getRecordingNames(train_test_string,mode):

    if mode == 'laosheng':
        recordings_train = ['lseh-Tan_Yang_jia-Hong_yang_dong-qm', 'lseh-Wei_guo_jia-Hong_yang_dong01-lon', 'lseh-Wei_guo_jia-Hong_yang_dong02-qm', 'lseh-Yi_lun_ming-Wen_zhao_guan-qm', 'lseh-Zi_na_ri-Hong_yang_dong-qm', 'lsfxp-Yang_si_lang-Si_lang_tan_mu-lon', 'lsxp-Huai_nan_wang-Huai_he_ying01-lon', 'lsxp-Jiang_shen_er-San_jia_dian02-qm', 'lsxp-Qian_bai_wan-Si_lang_tang_mu01-qm', 'lsxp-Quan_qian_sui-Gan_lu_si-qm', 'lsxp-Shen_gong_wu-Gan_lu_si-qm', 'lsxp-Shi_ye_shuo-Ding_jun_shan-qm', 'lsxp-Wo_ben_shi-Kong_cheng_ji-qm', 'lsxp-Wo_zheng_zai-Kong_cheng_ji01-upf', 'lsxp-Wo_zheng_zai-Kong_cheng_ji04-qm', 'lsxp-Xi_ri_you-Zhu_lian_zhai-qm']
        recordings_test  = ['lseh-Wo_ben_shi-Qiong_lin_yan-qm', 'lsxp-Guo_liao_yi-Wen_zhao_guan-qm', 'lsxp-Guo_liao_yi-Wen_zhao_guan-upf', 'lsxp-Huai_nan_wang-Huai_he_ying02-qm', 'lsxp-Jiang_shen_er-San_jia_dian01-1-upf', 'lsxp-Jiang_shen_er-San_jia_dian01-2-upf', 'lsxp-Wo_zheng_zai-Kong_cheng_ji02-qm']
    elif mode == 'dan':
        recordings_train = ['shiwenhui_tingxiongyan', 'wangjiangting_zhijianta', 'xixiangji_diyilai', 'xixiangji_luanchouduo', 'xixiangji_manmufeng', 'xixiangji_zhenmeijiu', 'yutangchun_yutangchun', 'zhuangyuanmei_daocishi', 'zhuangyuanmei_tianbofu', 'zhuangyuanmei_zhenzhushan', 'zhuangyuanmei_zinari']
        recordings_test  = ['wangjiangting_dushoukong', 'xixiangji_biyuntian', 'xixiangji_xianzhishuo', 'zhuangyuanmei_fudingkui']


    if train_test_string == 'TRAIN':
        name_recordings      = recordings_train
    else:
        name_recordings      = recordings_test
    return name_recordings

def getRecordingNamesSimi(train_test,mode):
    '''
    these recordings are in score corpus
    :param mode:
    :return:
    '''
    if mode == 'laosheng':
        recordings_train = ['lseh-Tan_Yang_jia-Hong_yang_dong-qm',
                            'lsfxp-Yang_si_lang-Si_lang_tan_mu-lon',
                            'lsxp-Qian_bai_wan-Si_lang_tang_mu01-qm',
                            'lsxp-Quan_qian_sui-Gan_lu_si-qm',
                            'lsxp-Shi_ye_shuo-Ding_jun_shan-qm',
                            'lsxp-Wo_zheng_zai-Kong_cheng_ji01-upf',
                            'lsxp-Wo_zheng_zai-Kong_cheng_ji04-qm',
                            'lsxp-Xi_ri_you-Zhu_lian_zhai-qm',
                            'lseh-Wo_ben_shi-Qiong_lin_yan-qm',
                            'lsxp-Guo_liao_yi-Wen_zhao_guan-qm',
                            'lsxp-Guo_liao_yi-Wen_zhao_guan-upf',
                            'lsxp-Huai_nan_wang-Huai_he_ying01-lon',
                            'lsxp-Wo_zheng_zai-Kong_cheng_ji02-qm']
        recordings_test = [
                      'lseh-Wei_guo_jia-Hong_yang_dong01-lon',
                      'lseh-Wei_guo_jia-Hong_yang_dong02-qm',
                      'lseh-Yi_lun_ming-Wen_zhao_guan-qm',
                      'lseh-Zi_na_ri-Hong_yang_dong-qm', # 4,5 not in corpus
                      'lsxp-Huai_nan_wang-Huai_he_ying02-qm', # 0,1,2,3 not in corpus
                      'lsxp-Jiang_shen_er-San_jia_dian01-1-upf',
                      'lsxp-Jiang_shen_er-San_jia_dian01-2-upf',
                      'lsxp-Jiang_shen_er-San_jia_dian02-qm',
                      'lsxp-Shen_gong_wu-Gan_lu_si-qm',
                      'lsxp-Wo_ben_shi-Kong_cheng_ji-qm']
    elif mode == 'dan':
        recordings_test = [
                      'zhuangyuanmei_zinari'
       ]

    if train_test == 'TRAIN':
        return recordings_train
    else:
        return recordings_test
