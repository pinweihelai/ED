# -*- coding: utf-8 -*-
import sys
sys.path.append('../hanlp/')

from . import hanlp
#from hanlp import NLPTool
nlpTool = hanlp.hanlp.NLPTool()

content = "你在哪里桐庐吗？"
params = {
    'enableCustomDic': True,
    'enablePOSTagging': True
}
print nlpTool.segment(content, params)['response']