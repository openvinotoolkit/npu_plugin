// Code generated by the FlatBuffers compiler. DO NOT EDIT.

package NamespaceA

import (
	flatbuffers "github.com/google/flatbuffers/go"
)

type SecondTableInA struct {
	_tab flatbuffers.Table
}

func GetRootAsSecondTableInA(buf []byte, offset flatbuffers.UOffsetT) *SecondTableInA {
	n := flatbuffers.GetUOffsetT(buf[offset:])
	x := &SecondTableInA{}
	x.Init(buf, n+offset)
	return x
}

func (rcv *SecondTableInA) Init(buf []byte, i flatbuffers.UOffsetT) {
	rcv._tab.Bytes = buf
	rcv._tab.Pos = i
}

func (rcv *SecondTableInA) Table() flatbuffers.Table {
	return rcv._tab
}

func (rcv *SecondTableInA) ReferToC(obj *TableInC) *TableInC {
	o := flatbuffers.UOffsetT(rcv._tab.Offset(4))
	if o != 0 {
		x := rcv._tab.Indirect(o + rcv._tab.Pos)
		if obj == nil {
			obj = new(TableInC)
		}
		obj.Init(rcv._tab.Bytes, x)
		return obj
	}
	return nil
}

func SecondTableInAStart(builder *flatbuffers.Builder) {
	builder.StartObject(1)
}
func SecondTableInAAddReferToC(builder *flatbuffers.Builder, referToC flatbuffers.UOffsetT) {
	builder.PrependUOffsetTSlot(0, flatbuffers.UOffsetT(referToC), 0)
}
func SecondTableInAEnd(builder *flatbuffers.Builder) flatbuffers.UOffsetT {
	return builder.EndObject()
}
