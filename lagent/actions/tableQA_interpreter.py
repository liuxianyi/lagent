import copy
import io
from contextlib import redirect_stdout
from typing import Any, Optional, Type, List

from lagent.actions.base_action import BaseAction, tool_api, BasePluginRuntime
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode


from transformers import AutoTokenizer, AutoModelForTableQuestionAnswering, AutoModelForSeq2SeqLM
import pandas as pd
import torch


# re
import re
def containsChinese(prompt: str):
    zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
    if zhPattern.search(prompt):
        return True
    return False

class TableQAPluginRuntime(BasePluginRuntime):
    """
        model from https://huggingface.co/google/tapas-large-finetuned-wtq
    """
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self, device):
        super().__init__(device=device)
        # tableQA 
        model_name = "google/tapas-base-finetuned-wikisql-supervised"
        self.tb_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tb_model = AutoModelForTableQuestionAnswering.from_pretrained(model_name)
        self.tb_model.to(device)
        self.tb_model.eval()
        self.device = device


    def run(self, table: pd.DataFrame, queries: List[str]):
        # table = pd.DataFrame.from_dict(table)
        if containsChinese(queries):
            tls_input_ids = self.tsl_tokenizer([queries], return_tensors='pt').to(self.device)
            tls_outputs_ids = self.tsl_model.generate(**tls_input_ids)
            enQueries = self.tsl_tokenizer.batch_decode(tls_outputs_ids, skip_special_tokens=True)[0]
            queries = enQueries
            print(queries)

        tb_inputs = self.tb_tokenizer(table=table, queries=[queries], padding="max_length", return_tensors="pt").to(self.device)
        tb_outputs = self.tb_model(**tb_inputs)

        tb_inputs = tb_inputs.to('cpu')
        predicted_answer_coordinates, predicted_aggregation_indices = self.tb_tokenizer.convert_logits_to_predictions(
            tb_inputs, tb_outputs.logits.cpu().detach(), tb_outputs.logits_aggregation.cpu().detach()
        )
        id2aggregation = {0: "无操作", 1: "求和", 2: "平均", 3: "计数"}
        # import pdb; pdb.set_trace()
        outTemplate = "TableQAInterpreter工具执行输出结果为：子表数据为[{}]，聚合操作为[{}]，最终的结果为[{}]。"
        for coordinates, predicted_agg_indice in zip(predicted_answer_coordinates, predicted_aggregation_indices):
            cell_values = []
            if len(coordinates) == 1:
                cell_value = table.iat[coordinates[0]].strip()
                cell_values.append(cell_value)
            else:
                for coordinate in coordinates:
                    cell_values.append(table.iat[coordinate].strip())
            # import pdb; pdb.set_trace()
            if len(cell_values) == 0:
                return "TableQAInterpreter工具未能计算出，请换用其他工具！"
            elif predicted_agg_indice !=0 and all([cell_value.strip().isdigit() for cell_value in cell_values]):
                str_cell_values = ", ".join(cell_values)
                cell_values = [int(cell_value) for cell_value in cell_values]
                final_anwer = ""
                if predicted_agg_indice == 1:
                    final_anwer += str(sum(cell_values))
                elif predicted_agg_indice == 2:
                    final_anwer += str(1.0 * sum(cell_values) / len(cell_values)) 
                elif predicted_agg_indice == 3:
                    final_anwer += str(len(cell_values))
                
                return outTemplate.format(str_cell_values, id2aggregation[predicted_agg_indice], final_anwer)
            else: 
                return outTemplate.format(", ".join(cell_values), id2aggregation[predicted_agg_indice], ", ".join(cell_values))

            
        
class TableQAInterpreter(BaseAction):
    """A Python executor that can execute Python scripts.

    Args:
        device (str): 
        description (dict, Optional): The description of the action. Defaults to ``None``.
        parser (Type[BaseParser]): The parser class to process the
            action's inputs and outputs. Defaults to :class:`JsonParser`.
        enable (bool, optional): Whether the action is enabled. Defaults to
            ``True``.
    """

    def __init__(self,
                 device: str,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True) -> None:
        super().__init__(description, parser, enable)
        # tableQARuntime
        device = torch.device(device)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.runtime = TableQAPluginRuntime(device)

    @tool_api
    def run(self, table_path: str, query: str) -> ActionReturn:
        """表格文档（*.csv,*.xls,*.xlsx）分析工具，通过输入表格的路径和问题，工具选择子表数据和聚合操作（例如，无操作、求和、平均、计数等）。

        Args:
            table_path (:class:`str`): 表格路径
            query (:class:`str`): 问题

        Returns:
            answer (:class:`str`): 问题的结果
        """

        if table_path.endswith('.csv'):
            table = pd.read_csv(filepath_or_buffer=table_path, sep=',', dtype='object')
        elif table_path.endswith('.xls') or table_path.endswith('.xlsx'):
            table = pd.read_excel(filepath_or_buffer=table_path, dtype='object')

        return self.runtime.run(table, query)
        # from func_timeout import FunctionTimedOut, func_set_timeout
        # self.runtime = GenericRuntime()
        # try:
        #     tool_return = func_set_timeout(self.timeout)(self._call)(command)
        # except FunctionTimedOut as e:
        # tool_return = ActionReturn(type=self.name)
        # tool_return.errmsg = repr(e)
        # tool_return.state = ActionStatusCode.API_ERROR

