# Superpixel Anything Overlap Check

- Paper: https://arxiv.org/abs/2509.12791
- Official repo: https://github.com/waldo-j/spam

## Trainable SPAM variants exposed in the official code

- `spam_ssn`
- `spam_resnet50`
- `spam_resnet101`
- `spam_mobilenetv3`

## Check Result

Проверка на "совпадений нет" не проходит в строгом виде.

### Exact overlaps with methods already present in this repository

- `slic`: SLIC is explicitly discussed in the SPAM paper as the classic regular superpixel baseline.
- `ssn`: The official SPAM repo states it is based on the pytorch implementation of SSN.

### Architectural overlaps

- `ssn`: SPAM inherits the SSN soft clustering core; this is architectural overlap even when the method name differs.

### Lineage-level / likely family overlaps

- `sin`: Likely lineage overlap with the non-iterative superpixel family cited by SPAM; our in-repo implementation is an internal approximation.
- `sp_fcn`: Likely lineage overlap with the SFCN/SSFCN family cited by SPAM; our in-repo implementation is not the official upstream model.

### Exact overlap with trainable SPAM variants

- No exact method-id overlap with the new SPAM trainable variants (`spam_ssn`, `spam_resnet50`, `spam_resnet101`, `spam_mobilenetv3`).

## Practical conclusion

- Adding SPAM training support is still useful.
- But it should not be presented as fully disjoint from prior repo work, because `SSN` and `SLIC` already overlap with SPAM paper context, and `SP-FCN` / `SIN` are close family-level matches.

