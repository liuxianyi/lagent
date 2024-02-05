# flake8: noqa: E501
import copy
import io
from contextlib import redirect_stdout
from typing import Any, Optional, Type

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode


class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(
            self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)


class Image2VideoInterpreter(BaseAction):
    """A Python executor that can execute Python scripts.

    Args:
        answer_symbol (str, Optional): the answer symbol from LLM. Defaults to ``None``.
        answer_expr (str, Optional): the answer function name of the Python
            script. Defaults to ``'solution()'``.
        answer_from_stdout (boolean, Optional): whether the execution results is from
            stdout. Defaults to ``False``.
        timeout (int, Optional): Upper bound of waiting time for Python script execution.
            Defaults to ``20``.
        description (dict, Optional): The description of the action. Defaults to ``None``.
        parser (Type[BaseParser]): The parser class to process the
            action's inputs and outputs. Defaults to :class:`JsonParser`.
        enable (bool, optional): Whether the action is enabled. Defaults to
            ``True``.
    """

    def __init__(self,
                 device,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True) -> None:
        super().__init__(description, parser, enable)

    @tool_api
    def run(self, image_path: str) -> ActionReturn:
        """这是一种将图像生成视频的插件，它以静态图像作为输入，通过改工具直接生成视频。

        Args:
            image_path (:class:`str`): 静态图片路径

        Returns:
            video_path (:class:`str`): 生成的视频路径
        """
        # from func_timeout import FunctionTimedOut, func_set_timeout
        # self.runtime = GenericRuntime()
        # try:
        #     tool_return = func_set_timeout(self.timeout)(self._call)(command)
        # except FunctionTimedOut as e:
        # tool_return = ActionReturn(type=self.name)
        # tool_return.errmsg = repr(e)
        # tool_return.state = ActionStatusCode.API_ERROR
        return {"video_path": "/root/code/image.mp4"}

    def _call(self, command: str) -> ActionReturn:
        tool_return = ActionReturn(type=self.name)
        try:
            if '```python' in command:
                command = command.split('```python')[1].split('```')[0]
            elif '```' in command:
                command = command.split('```')[1].split('```')[0]
            tool_return.args = dict(text='```python\n' + command + '\n```')
            command = command.split('\n')

            if self.answer_from_stdout:
                program_io = io.StringIO()
                with redirect_stdout(program_io):
                    self.runtime.exec_code('\n'.join(command))
                program_io.seek(0)
                res = program_io.readlines()[-1]
            elif self.answer_symbol:
                self.runtime.exec_code('\n'.join(command))
                res = self.runtime._global_vars[self.answer_symbol]
            elif self.answer_expr:
                self.runtime.exec_code('\n'.join(command))
                res = self.runtime.eval_code(self.answer_expr)
            else:
                self.runtime.exec_code('\n'.join(command[:-1]))
                res = self.runtime.eval_code(command[-1])
        except Exception as e:
            tool_return.errmsg = repr(e)
            tool_return.type = self.name
            tool_return.state = ActionStatusCode.API_ERROR
            return tool_return
        try:
            tool_return.result = [dict(type='text', content=str(res))]
            tool_return.state = ActionStatusCode.SUCCESS
        except Exception as e:
            tool_return.errmsg = repr(e)
            tool_return.type = self.name
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return
