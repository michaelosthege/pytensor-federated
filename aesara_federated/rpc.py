# Generated by the protocol buffer compiler.  DO NOT EDIT!
# sources: service.proto
# plugin: python-betterproto
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    AsyncIterable,
    AsyncIterator,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)

import betterproto
import grpclib
from betterproto.grpc.grpclib_server import ServiceBase

from . import npproto


if TYPE_CHECKING:
    import grpclib.server
    from betterproto.grpc.grpclib_client import MetadataLike
    from grpclib.metadata import Deadline


@dataclass(eq=False, repr=False)
class InputArrays(betterproto.Message):
    """Input type message of the ArraysToArraysService"""

    items: List["npproto.Ndarray"] = betterproto.message_field(1)
    """A sequence of NumPy arrays"""

    uuid: str = betterproto.string_field(2)
    """A unique identifier of this input message."""


@dataclass(eq=False, repr=False)
class OutputArrays(betterproto.Message):
    """Output type message of the ArraysToArraysService"""

    items: List["npproto.Ndarray"] = betterproto.message_field(1)
    """A sequence of NumPy arrays"""

    uuid: str = betterproto.string_field(2)
    """The unique identifier of the corresponding input message."""


class ArraysToArraysServiceStub(betterproto.ServiceStub):
    async def evaluate(
        self,
        input_arrays: "InputArrays",
        *,
        timeout: Optional[float] = None,
        deadline: Optional["Deadline"] = None,
        metadata: Optional["MetadataLike"] = None
    ) -> "OutputArrays":
        return await self._unary_unary(
            "/ArraysToArraysService/Evaluate",
            input_arrays,
            OutputArrays,
            timeout=timeout,
            deadline=deadline,
            metadata=metadata,
        )

    async def evaluate_stream(
        self,
        input_arrays_iterator: Union[
            AsyncIterable["InputArrays"], Iterable["InputArrays"]
        ],
        *,
        timeout: Optional[float] = None,
        deadline: Optional["Deadline"] = None,
        metadata: Optional["MetadataLike"] = None
    ) -> AsyncIterator["OutputArrays"]:
        async for response in self._stream_stream(
            "/ArraysToArraysService/EvaluateStream",
            input_arrays_iterator,
            InputArrays,
            OutputArrays,
            timeout=timeout,
            deadline=deadline,
            metadata=metadata,
        ):
            yield response


class ArraysToArraysServiceBase(ServiceBase):
    async def evaluate(self, input_arrays: "InputArrays") -> "OutputArrays":
        raise grpclib.GRPCError(grpclib.const.Status.UNIMPLEMENTED)

    async def evaluate_stream(
        self, input_arrays_iterator: AsyncIterator["InputArrays"]
    ) -> AsyncIterator["OutputArrays"]:
        raise grpclib.GRPCError(grpclib.const.Status.UNIMPLEMENTED)

    async def __rpc_evaluate(
        self, stream: "grpclib.server.Stream[InputArrays, OutputArrays]"
    ) -> None:
        request = await stream.recv_message()
        response = await self.evaluate(request)
        await stream.send_message(response)

    async def __rpc_evaluate_stream(
        self, stream: "grpclib.server.Stream[InputArrays, OutputArrays]"
    ) -> None:
        request = stream.__aiter__()
        await self._call_rpc_handler_server_stream(
            self.evaluate_stream,
            stream,
            request,
        )

    def __mapping__(self) -> Dict[str, grpclib.const.Handler]:
        return {
            "/ArraysToArraysService/Evaluate": grpclib.const.Handler(
                self.__rpc_evaluate,
                grpclib.const.Cardinality.UNARY_UNARY,
                InputArrays,
                OutputArrays,
            ),
            "/ArraysToArraysService/EvaluateStream": grpclib.const.Handler(
                self.__rpc_evaluate_stream,
                grpclib.const.Cardinality.STREAM_STREAM,
                InputArrays,
                OutputArrays,
            ),
        }
