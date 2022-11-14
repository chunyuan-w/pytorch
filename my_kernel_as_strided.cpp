#include <dlfcn.h>
#include <assert.h>
#include <chrono>
#include <python3.7/Python.h>
#include <torch/torch.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>

class LoadKernel_call0{
  public:
    LoadKernel_call0() {
    auto kernel0_lib = dlopen("/tmp/torchinductor_chunyuan/lp/clp6m37zeydchk676vnaypzwv6w7cmpuz3vdqmmntoglax7o2unh.so", RTLD_NOW);
    assert(kernel0_lib != nullptr);
    *(void **) (&kernel0) = dlsym(kernel0_lib, "kernel");

}
void (*kernel0)(float*,const float*);

};

    static PyObject* call_0(PyObject* dummy, PyObject* args)  {

        at::Tensor arg0_1;


        std::vector<at::Tensor> checks;
        if (!PyTuple_CheckExact(args)) {
            PyErr_SetString(PyExc_TypeError, "expected tuple()");
            return NULL;
        }
        auto len = PyTuple_GET_SIZE(args);
        checks.reserve(len);
        for (auto i : c10::irange(len)) {
            PyObject* item = PyTuple_GET_ITEM(args, i);
            if (!PyList_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "expected list");
            return NULL;
            }

            auto len_list = PyList_GET_SIZE(item);
            for (auto i : c10::irange(len_list)) {
            PyObject* tensor = PyList_GET_ITEM(item, i);
            if (!THPVariable_CheckExact(tensor) && !THPVariable_Check(tensor)) {
            PyErr_SetString(PyExc_TypeError, "expected Tensor()");
            return NULL;
            }
            arg0_1 = THPVariable_Unpack(tensor);
            }


            // clear args
            if (PyList_SetSlice(item, 0, PyList_Size(item), NULL)<0) {
                PyErr_SetString(PyExc_TypeError, "PyList_SetSlice Failed");
                return NULL;
            }


        }


        static LoadKernel_call0 load_kernel_;
        auto buf0 = at::empty_strided({64, 64}, {64, 1}, at::ScalarType::Float); 
        auto buf1 = at::as_strided(buf0, {8, 8, 64}, {512, 64, 1}); buf0.reset();  // reuse
        load_kernel_.kernel0((float*)(buf1.data_ptr()), (float*)(arg0_1.data_ptr()));
        return torch::autograd::utils::wrap(std::vector<at::Tensor>({at::as_strided(arg0_1, {8, 8, 64}, {512, 64, 1}), buf1})); 
    }



    static PyMethodDef moduleMethods[] = {
        {"call_0",  call_0, METH_VARARGS,
        "Execute a shell command."},
        {NULL, NULL, 0, NULL}        /* Sentinel */
    };

    static struct PyModuleDef _module =
        {PyModuleDef_HEAD_INIT, "_module", "", -1, moduleMethods};


    PyMODINIT_FUNC PyInit__module(void)
    {
        return PyModule_Create(&_module);
    }
