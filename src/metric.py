# Copyright 2020 IBM
# Author: peter.zhong@au1.ibm.com
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the Apache 2.0 License.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Apache 2.0 License for more details.
import numpy as np
import distance
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from collections import deque
from parallel import parallel_process
from tqdm import tqdm

class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.


class TEDS(object):
    ''' Tree Edit Distance basead Similarity
    '''
    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (n_jobs >= 1), 'n_jobs must be an integer greather than 1'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        ''' Converts HTML tree to the format required by apted
        '''
        global __tokens__
        if node.tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(node.tag,
                                 int(node.attrib.get('colspan', '1')),
                                 int(node.attrib.get('rowspan', '1')),
                                 cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        ''' Computes TEDS score between the prediction and the ground truth of a
            given sample
        '''
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath('body/table') and true.xpath('body/table'):
            pred = pred.xpath('body/table')[0]
            true = true.xpath('body/table')[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0

    def batch_evaluate(self, pred_json, true_json):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
            @params pred_json: {'FILENAME': 'HTML CODE', ...}
            @params true_json: {'FILENAME': {'html': 'HTML CODE'}, ...}
            @output: {'FILENAME': 'TEDS SCORE', ...}
        '''
        samples = true_json.keys()
        if self.n_jobs == 1:
            scores = [self.evaluate(pred_json.get(filename, ''), true_json[filename]['html']) for filename in tqdm(samples)]
        else:
            inputs = [{'pred': pred_json.get(filename, ''), 'true': true_json[filename]['html']} for filename in samples]
            scores = parallel_process(inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1)
        scores = dict(zip(samples, scores))
        return scores


if __name__ == '__main__':
    import json
    import pprint
    with open('rr_pred.json') as fp:
        pred_json = json.load(fp)
    with open('rr_gt.json') as fp:
        true_json = json.load(fp)
    teds = TEDS(n_jobs=4)
    scores = teds.batch_evaluate(pred_json, true_json)
    pp = pprint.PrettyPrinter()
    pp.pprint(scores)
    #_______________________________________________________________________
    # scores = {'PMC1277835_006_00.png': 0.0,
    #           'PMC1397844_003_00.png': 0.0,
    #          'PMC1657033_005_00.png': 0.922320334992801,
    #          'PMC1876796_002_01.png': 0.9398551150430561,
    #          'PMC1892548_003_01.png': 0.0,
    #          'PMC2140266_002_00.png': 0.0,
    #          'PMC2262879_003_00.png': 0.0,
    #          'PMC2438334_002_00.png': 0.9369825205171031,
    #          'PMC2864246_002_00.png': 0.0,
    #          'PMC2984508_007_00.png': 0.9048789408915064,
    #          'PMC2994852_003_00.png': 0.0,
    #          'PMC3001744_003_01.png': 0.8479690933028248,
    #          'PMC3135498_002_00.png': 0.0,
    #          'PMC3234574_005_01.png': 0.9569592952754014,
    #          'PMC3269686_002_00.png': 0.8770728770728771,
    #          'PMC3284444_003_00.png': 0.0,
    #          'PMC3305515_006_00.png': 0.0,
    #          'PMC3310274_002_01.png': 0.9034682690280413,
    #          'PMC3315433_005_00.png': 0.8714407797414814,
    #          'PMC3359276_016_00.png': 0.8501000350082193,
    #          'PMC3395581_002_00.png': 0.8384544219266907,
    #          'PMC3495651_007_01.png': 0.0,
    #          'PMC3528621_004_00.png': 0.9352165114480505,
    #          'PMC3529481_001_00.png': 0.9396787161226974,
    #          'PMC3545907_005_00.png': 0.0,
    #          'PMC3548710_004_01.png': 0.8950262110044637,
    #          'PMC3565896_004_00.png': 0.0,
    #          'PMC3583133_006_00.png': 0.0,
    #          'PMC3585467_003_01.png': 0.873847978936753,
    #          'PMC3602097_010_00.png': 0.0,
    #          'PMC3709295_002_00.png': 0.8657842843566534,
    #          'PMC3762796_002_00.png': 0.0,
    #          'PMC3791805_008_02.png': 0.9355118430024921,
    #          'PMC3821366_020_00.png': 0.0,
    #          'PMC3860015_004_02.png': 0.0,
    #          'PMC3899611_007_02.png': 0.0,
    #          'PMC3953622_002_02.png': 0.0,
    #          'PMC3971348_007_00.png': 0.9083119216318323,
    #          'PMC4001017_001_00.png': 0.8893506493506493,
    #          'PMC4001112_001_00.png': 0.0,
    #          'PMC4028858_009_00.png': 0.8960695765658908,
    #          'PMC4029345_007_00.png': 0.0,
    #          'PMC4039985_003_00.png': 0.7728955628796708,
    #          'PMC4047779_002_00.png': 0.9125106467568878,
    #          'PMC4056361_001_00.png': 0.0,
    #          'PMC4065580_002_00.png': 0.0,
    #          'PMC4089935_005_00.png': 0.9645842904484505,
    #          'PMC4127295_003_00.png': 0.0,
    #          'PMC4183102_001_00.png': 0.0,
    #          'PMC4216444_004_00.png': 0.7836324433648626,
    #          'PMC4246451_002_00.png': 0.7194132894044438,
    #          'PMC4288364_003_01.png': 0.0,
    #          'PMC4451338_008_00.png': 0.0,
    #          'PMC4462181_004_01.png': 0.0,
    #          'PMC4517657_002_00.png': 0.0,
    #          'PMC4585417_003_02.png': 0.9192339460761848,
    #          'PMC4632410_004_00.png': 0.0,
    #          'PMC4763428_002_00.png': 0.8643696861539467,
    #          'PMC4764644_010_00.png': 0.838543538102129,
    #          'PMC4780114_005_00.png': 0.0,
    #          'PMC4823688_003_00.png': 0.9424819670514742,
    #          'PMC4854880_002_00.png': 0.9139807504340132,
    #          'PMC4898376_005_01.png': 0.0,
    #          'PMC4930793_005_00.png': 0.7913450128613247,
    #          'PMC4938925_007_00.png': 0.891313929877318,
    #          'PMC4945830_003_00.png': 0.0,
    #          'PMC4947861_005_00.png': 0.0,
    #          'PMC4980799_003_00.png': 0.9425665828132732,
    #          'PMC4999787_007_00.png': 0.0,
    #          'PMC5020053_002_01.png': 0.8047084258749856,
    #          'PMC5036701_007_00.png': 0.0,
    #          'PMC5108870_006_00.png': 0.0,
    #          'PMC5156306_003_00.png': 0.917305635675372,
    #          'PMC5294978_003_00.png': 0.0,
    #          'PMC5302276_005_00.png': 0.0,
    #          'PMC5312441_006_00.png': 0.0,
    #          'PMC5420145_004_00.png': 0.0,
    #          'PMC5504338_003_00.png': 0.0,
    #          'PMC5530919_002_01.png': 0.0,
    #          'PMC5576415_008_00.png': 0.8994265288969713,
    #          'PMC5618660_001_00.png': 0.888303783439915,
    #          'PMC5636747_005_00.png': 0.0,
    #          'PMC5725644_006_00.png': 0.0,
    #          'PMC5778640_004_00.png': 0.9411118352497824,
    #          'PMC5795900_016_00.png': 0.0,
    #          'PMC5813361_005_01.png': 0.0,
    #          'PMC5840258_004_00.png': 0.8210181451612903,
    #          'PMC5846843_004_00.png': 0.0,
    #          'PMC5852766_014_01.png': 0.0,
    #          'PMC5874598_008_00.png': 0.0,
    #          'PMC5874691_003_00.png': 0.0,
    #          'PMC5968540_007_00.png': 0.0,
    #          'PMC5978174_003_01.png': 0.8419710289545657}
    # scores = {'PMC1277835_006_00.png': 0.8915212238741651,
    #              'PMC1397844_003_00.png': 0.9244154972763442,
    #              'PMC1657033_005_00.png': 0.922320334992801,
    #              'PMC1876796_002_01.png': 0.9398551150430561,
    #              'PMC1892548_003_01.png': 0.9398583780969678,
    #              'PMC2140266_002_00.png': 0.9810637511817015,
    #              'PMC2262879_003_00.png': 0.9504678562699889,
    #              'PMC2438334_002_00.png': 0.9369825205171031,
    #              'PMC2864246_002_00.png': 0.9304026910275389,
    #              'PMC2984508_007_00.png': 0.9048789408915064,
    #              'PMC2994852_003_00.png': 0.9007943348358816,
    #              'PMC3001744_003_01.png': 0.8479690933028248,
    #              'PMC3135498_002_00.png': 0.8154044748027196,
    #              'PMC3234574_005_01.png': 0.9569592952754014,
    #              'PMC3269686_002_00.png': 0.8770728770728771,
    #              'PMC3284444_003_00.png': 0.824315960486976,
    #              'PMC3305515_006_00.png': 0.9469129115909446,
    #              'PMC3310274_002_01.png': 0.9034682690280413,
    #              'PMC3315433_005_00.png': 0.8714407797414814,
    #              'PMC3359276_016_00.png': 0.8501000350082193,
    #              'PMC3395581_002_00.png': 0.8384544219266907,
    #              'PMC3495651_007_01.png': 0.8473921839073232,
    #              'PMC3528621_004_00.png': 0.9352165114480505,
    #              'PMC3529481_001_00.png': 0.9396787161226974,
    #              'PMC3545907_005_00.png': 0.8623672749124659,
    #              'PMC3548710_004_01.png': 0.8950262110044637,
    #              'PMC3565896_004_00.png': 0.8860428438149309,
    #              'PMC3583133_006_00.png': 0.8016527791349349,
    #              'PMC3585467_003_01.png': 0.873847978936753,
    #              'PMC3602097_010_00.png': 0.827479887549687,
    #              'PMC3709295_002_00.png': 0.8657842843566534,
    #              'PMC3762796_002_00.png': 0.8726482545552421,
    #              'PMC3791805_008_02.png': 0.9355118430024921,
    #              'PMC3821366_020_00.png': 0.8642739867692429,
    #              'PMC3860015_004_02.png': 0.9001116415296483,
    #              'PMC3899611_007_02.png': 0.8587325753007952,
    #              'PMC3953622_002_02.png': 0.8510883801460237,
    #              'PMC3971348_007_00.png': 0.9083119216318323,
    #              'PMC4001017_001_00.png': 0.8893506493506493,
    #              'PMC4001112_001_00.png': 0.8139941799448283,
    #              'PMC4028858_009_00.png': 0.8960695765658908,
    #              'PMC4029345_007_00.png': 0.9220237424481098,
    #              'PMC4039985_003_00.png': 0.7728955628796708,
    #              'PMC4047779_002_00.png': 0.9125106467568878,
    #              'PMC4056361_001_00.png': 0.8196575927477168,
    #              'PMC4065580_002_00.png': 0.9149303718686845,
    #              'PMC4089935_005_00.png': 0.9645842904484505,
    #              'PMC4127295_003_00.png': 0.8690465499383169,
    #              'PMC4183102_001_00.png': 0.8189337477396009,
    #              'PMC4216444_004_00.png': 0.7836324433648626,
    #              'PMC4246451_002_00.png': 0.7194132894044438,
    #              'PMC4288364_003_01.png': 0.9194674880874336,
    #              'PMC4451338_008_00.png': 0.8456198237593586,
    #              'PMC4462181_004_01.png': 0.8840852235598109,
    #              'PMC4517657_002_00.png': 0.9105789619185383,
    #              'PMC4585417_003_02.png': 0.9192339460761848,
    #              'PMC4632410_004_00.png': 0.880572606200006,
    #              'PMC4763428_002_00.png': 0.8643696861539467,
    #              'PMC4764644_010_00.png': 0.838543538102129,
    #              'PMC4780114_005_00.png': 0.9143758649738458,
    #              'PMC4823688_003_00.png': 0.9424819670514742,
    #              'PMC4854880_002_00.png': 0.9139807504340132,
    #              'PMC4898376_005_01.png': 0.8812093036434282,
    #              'PMC4930793_005_00.png': 0.7913450128613247,
    #              'PMC4938925_007_00.png': 0.891313929877318,
    #              'PMC4945830_003_00.png': 0.9527804682391795,
    #              'PMC4947861_005_00.png': 0.940105540586927,
    #              'PMC4980799_003_00.png': 0.9425665828132732,
    #              'PMC4999787_007_00.png': 0.8016714843234378,
    #              'PMC5020053_002_01.png': 0.8047084258749856,
    #              'PMC5036701_007_00.png': 0.9622932874425915,
    #              'PMC5108870_006_00.png': 0.9522660690962785,
    #              'PMC5156306_003_00.png': 0.917305635675372,
    #              'PMC5294978_003_00.png': 0.9338036890578882,
    #              'PMC5302276_005_00.png': 0.9132269004616675,
    #              'PMC5312441_006_00.png': 0.8463247917917972,
    #              'PMC5420145_004_00.png': 0.8519053229825669,
    #              'PMC5504338_003_00.png': 0.8812043716702812,
    #              'PMC5530919_002_01.png': 0.9302910983879458,
    #              'PMC5576415_008_00.png': 0.8994265288969713,
    #              'PMC5618660_001_00.png': 0.888303783439915,
    #              'PMC5636747_005_00.png': 0.8805678241316539,
    #              'PMC5725644_006_00.png': 0.8151435335149202,
    #              'PMC5778640_004_00.png': 0.9411118352497824,
    #              'PMC5795900_016_00.png': 0.8666228558823819,
    #              'PMC5813361_005_01.png': 0.826914722868452,
    #              'PMC5840258_004_00.png': 0.8210181451612903,
    #              'PMC5846843_004_00.png': 0.9202606425451035,
    #              'PMC5852766_014_01.png': 0.9805874272676229,
    #              'PMC5874598_008_00.png': 0.9279330227124226,
    #              'PMC5874691_003_00.png': 0.8242588425001652,
    #              'PMC5968540_007_00.png': 0.943030876017033,
    #              'PMC5978174_003_01.png': 0.8419710289545657}
    # sorted_scores = [('PMC4246451_002_00.png', 0.7194132894044438), ('PMC4039985_003_00.png', 0.7728955628796708), ('PMC4216444_004_00.png', 0.7836324433648626), ('PMC4930793_005_00.png', 0.7913450128613247), ('PMC3583133_006_00.png', 0.8016527791349349), ('PMC4999787_007_00.png', 0.8016714843234378), ('PMC5020053_002_01.png', 0.8047084258749856), ('PMC4001112_001_00.png', 0.8139941799448283), ('PMC5725644_006_00.png', 0.8151435335149202), ('PMC3135498_002_00.png', 0.8154044748027196), ('PMC4183102_001_00.png', 0.8189337477396009), ('PMC4056361_001_00.png', 0.8196575927477168), ('PMC5840258_004_00.png', 0.8210181451612903), ('PMC5874691_003_00.png', 0.8242588425001652), ('PMC3284444_003_00.png', 0.824315960486976), ('PMC5813361_005_01.png', 0.826914722868452), ('PMC3602097_010_00.png', 0.827479887549687), ('PMC3395581_002_00.png', 0.8384544219266907), ('PMC4764644_010_00.png', 0.838543538102129), ('PMC5978174_003_01.png', 0.8419710289545657), ('PMC4451338_008_00.png', 0.8456198237593586), ('PMC5312441_006_00.png', 0.8463247917917972), ('PMC3495651_007_01.png', 0.8473921839073232), ('PMC3001744_003_01.png', 0.8479690933028248), ('PMC3359276_016_00.png', 0.8501000350082193), ('PMC3953622_002_02.png', 0.8510883801460237), ('PMC5420145_004_00.png', 0.8519053229825669), ('PMC3899611_007_02.png', 0.8587325753007952), ('PMC3545907_005_00.png', 0.8623672749124659), ('PMC3821366_020_00.png', 0.8642739867692429), ('PMC4763428_002_00.png', 0.8643696861539467), ('PMC3709295_002_00.png', 0.8657842843566534), ('PMC5795900_016_00.png', 0.8666228558823819), ('PMC4127295_003_00.png', 0.8690465499383169), ('PMC3315433_005_00.png', 0.8714407797414814), ('PMC3762796_002_00.png', 0.8726482545552421), ('PMC3585467_003_01.png', 0.873847978936753), ('PMC3269686_002_00.png', 0.8770728770728771), ('PMC5636747_005_00.png', 0.8805678241316539), ('PMC4632410_004_00.png', 0.880572606200006), ('PMC5504338_003_00.png', 0.8812043716702812), ('PMC4898376_005_01.png', 0.8812093036434282), ('PMC4462181_004_01.png', 0.8840852235598109), ('PMC3565896_004_00.png', 0.8860428438149309), ('PMC5618660_001_00.png', 0.888303783439915), ('PMC4001017_001_00.png', 0.8893506493506493), ('PMC4938925_007_00.png', 0.891313929877318), ('PMC1277835_006_00.png', 0.8915212238741651), ('PMC3548710_004_01.png', 0.8950262110044637), ('PMC4028858_009_00.png', 0.8960695765658908), ('PMC5576415_008_00.png', 0.8994265288969713), ('PMC3860015_004_02.png', 0.9001116415296483), ('PMC2994852_003_00.png', 0.9007943348358816), ('PMC3310274_002_01.png', 0.9034682690280413), ('PMC2984508_007_00.png', 0.9048789408915064), ('PMC3971348_007_00.png', 0.9083119216318323), ('PMC4517657_002_00.png', 0.9105789619185383), ('PMC4047779_002_00.png', 0.9125106467568878), ('PMC5302276_005_00.png', 0.9132269004616675), ('PMC4854880_002_00.png', 0.9139807504340132), ('PMC4780114_005_00.png', 0.9143758649738458), ('PMC4065580_002_00.png', 0.9149303718686845), ('PMC5156306_003_00.png', 0.917305635675372), ('PMC4585417_003_02.png', 0.9192339460761848), ('PMC4288364_003_01.png', 0.9194674880874336), ('PMC5846843_004_00.png', 0.9202606425451035), ('PMC4029345_007_00.png', 0.9220237424481098), ('PMC1657033_005_00.png', 0.922320334992801), ('PMC1397844_003_00.png', 0.9244154972763442), ('PMC5874598_008_00.png', 0.9279330227124226), ('PMC5530919_002_01.png', 0.9302910983879458), ('PMC2864246_002_00.png', 0.9304026910275389), ('PMC5294978_003_00.png', 0.9338036890578882), ('PMC3528621_004_00.png', 0.9352165114480505), ('PMC3791805_008_02.png', 0.9355118430024921), ('PMC2438334_002_00.png', 0.9369825205171031), ('PMC3529481_001_00.png', 0.9396787161226974), ('PMC1876796_002_01.png', 0.9398551150430561), ('PMC1892548_003_01.png', 0.9398583780969678), ('PMC4947861_005_00.png', 0.940105540586927), ('PMC5778640_004_00.png', 0.9411118352497824), ('PMC4823688_003_00.png', 0.9424819670514742), ('PMC4980799_003_00.png', 0.9425665828132732), ('PMC5968540_007_00.png', 0.943030876017033), ('PMC3305515_006_00.png', 0.9469129115909446), ('PMC2262879_003_00.png', 0.9504678562699889), ('PMC5108870_006_00.png', 0.9522660690962785), ('PMC4945830_003_00.png', 0.9527804682391795), ('PMC3234574_005_01.png', 0.9569592952754014), ('PMC5036701_007_00.png', 0.9622932874425915), ('PMC4089935_005_00.png', 0.9645842904484505), ('PMC5852766_014_01.png', 0.9805874272676229), ('PMC2140266_002_00.png', 0.9810637511817015)]

    np_arr = np.array(list(scores.values()))
    # sorted(scores.items(), key = lambda kv:(kv[1], kv[0]))
    # print(sorted(scores.items(), key = lambda kv:(kv[1], kv[0])))
    exist = (np_arr != 0)
    num = np_arr.sum()
    den = exist.sum()
    print(max(np_arr))
    print(min(np_arr))
    print(np.median(np_arr))
    print(np.mean(np_arr))
    print(num / den)
