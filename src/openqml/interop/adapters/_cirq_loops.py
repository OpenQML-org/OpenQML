"""Native Cirq repeat-until forms that retain the final classical record."""

from dataclasses import replace

from ...ir.parameters import Expression


def repeat_forms(circuit):
    forms, written = {}, set()
    for index, op in enumerate(circuit.operations):
        if op.name == "while_loop":
            body = op.blocks[0]
            condition, inverted = op.condition, True
            if condition is None and body.operations:
                end = body.operations[-1]
                if end.name == "if_else":
                    yes, no = end.blocks
                    if (
                        len(yes.operations) == 1
                        and yes.operations[0].name == "break_loop"
                        and not no.operations
                        and yes.global_phase == no.global_phase == Expression()
                    ):
                        body = replace(body, operations=tuple(body.operations[:-1]))
                        condition, inverted = end.condition, False
            measured = {item.clbits[0] for item in body.operations if item.name == "measure"}
            if (
                condition is not None
                and len(condition.bits) <= 10
                and measured.issuperset(condition.bits)
                and not body.has_runtime_control
                and (not inverted or (condition.value == 0 and written.isdisjoint(condition.bits)))
            ):
                forms[index] = body, condition, inverted
        written.update(op.clbits)
        for block in op.blocks:
            written.update(sub_bit for _, sub in block.walk_operations() for sub_bit in sub.clbits)
    return forms
