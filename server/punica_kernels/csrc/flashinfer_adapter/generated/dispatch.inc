#define _DISPATCH_CASES_page_size(...)       \
  _DISPATCH_CASE(16, PAGE_SIZE, __VA_ARGS__) \
// EOL
#define _DISPATCH_CASES_group_size(...)      \
  _DISPATCH_CASE(1, GROUP_SIZE, __VA_ARGS__) \
  _DISPATCH_CASE(2, GROUP_SIZE, __VA_ARGS__) \
  _DISPATCH_CASE(4, GROUP_SIZE, __VA_ARGS__) \
  _DISPATCH_CASE(8, GROUP_SIZE, __VA_ARGS__) \
// EOL
#define _DISPATCH_CASES_head_dim(...)        \
  _DISPATCH_CASE(128, HEAD_DIM, __VA_ARGS__) \
// EOL

