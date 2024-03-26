codebase:
	rm -rf build
	mkdir build	
	cp -r third_party/text-generation-inference/benchmark build/
	cp -r third_party/text-generation-inference/clients build/
	cp -r third_party/text-generation-inference/integration-tests build/
	cp -r third_party/text-generation-inference/launcher build/
	cp -r third_party/text-generation-inference/load_tests build/
	cp -r third_party/text-generation-inference/proto build/
	cp -r third_party/text-generation-inference/router build/
	cp -r third_party/text-generation-inference/server build/
	cp third_party/text-generation-inference/Cargo*.* build/
	cp -r server build/

install-server:
	cd build/server && make install
	cd punica_kernels && pip install -v --no-build-isolation .

install-custom-kernels:
	if [ "$$BUILD_EXTENSIONS" = "True" ]; then cd build/server/custom_kernels && python setup.py install; else echo "Custom kernels are disabled, you need to set the BUILD_EXTENSIONS environment variable to 'True' in order to build them. (Please read the docs, kernels might not work on all hardware)"; fi

install-router:
	cd build/router && cargo install --path .

install-launcher:
	cd build/launcher && cargo install --path .

install-benchmark:
	cd build/benchmark && cargo install --path .

install: codebase install-server install-router install-launcher install-custom-kernels
