#!/bin/bash
# Script that requires to be run inside the docker image

# Exit on error
set -e
# Exit on unset variables
# set -u

fatal() {
    _FATAL="\033[0;41m[FATAL]\033[0m"
    (>&2 echo -e "${_FATAL} $*")
    exit 1
}

info() {
    _INFO="\033[0;32m[INFO]\033[0m"
    echo -e "${_INFO} $*"
}


cmd_help() {
cat<<EOF
Usage: $0 [command] [--options ...]

Description:
  Handy bash script to build openpgl and its dependencies (tbb + pytorch)
  (BEFORE) installing renderman. Note that openpgl is build statically
  but torch dependencies will be dynamic.

Commands:
  build             Build openpgl and its dependencies.
  clean             Clean build artifacts.

Global options:
  --docker	    run outside docker environment (default: 1)

  help, -h, --help  Show this help.

EOF
}

cmd_clean() {
    info "Cleaning up all artifacts"
    rm -rvf build install
    info "Done."
}

_check_if_inside_container() {
    if [[ ! -f /.dockerenv ]]; then
        fatal "Should be run inside the container."
    fi
}

_check_status() {
    read -e -p "Continue? [y/N]: " yN
    case $yN in
	[Yy]*)
	    return 0
	    ;;
	*)
	    fatal "Aborted"
	    ;;
    esac
}

cmd_build() {

    if [[ ${DOCKER} == 1 ]]; then
	# Activate devtoolset
	source /opt/rh/devtoolset-9/enable
    fi

    info "Building openpgl..."

    # We build both in release mode!
    CMAKE_BUILD_TYPE=RelWithDebInfo


    info "Summary:"
    info "gcc        : $(which gcc)"
    info "build type : ${CMAKE_BUILD_TYPE}"

    # Disable neural components: torch, eigen
    # environment variable OPENPGL_BUILD_NEURAL_COMPONENT=1
    CMAKE_FLAGS="-DCMAKE_INSTALL_PREFIX=$(realpath install)"
    CMAKE_FLAGS="${CMAKE_FLAGS} -DBUILD_PYTORCH=OFF"
    CMAKE_FLAGS="${CMAKE_FLAGS} -DBUILD_NEURAL_COMPONENTS=OFF"

    # Configure
    info "configuring..."
    info "cmake -B build -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} ${CMAKE_FLAGS} superbuild/"

    cmake -B build -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} ${CMAKE_FLAGS} superbuild

    # Building library
    info "building..."
    cmake --build build --parallel

    # Install
    info "installing..."
    cmake --install build

    info "Done."
}


main() {
    if [[ $# == 0 || $1 == "-h" || $1 == "--help" || $1 == "help" ]]; then
	cmd_help
	exit 0
    fi

    DOCKER=1
    # parse options
    args=$(getopt -l "docker:" -- "$0" "$@")
    eval set -- "${args}"
    while [[ $# -ne 0 ]]; do
	case $1 in
	    --docker)
		DOCKER="$2"
		shift 2
		;;
	    --)
		shift
		break
		;;
	esac
    done

    if [[ ${DOCKER} == 1 ]]; then
	info "Building inside docker..."
	_check_if_inside_container
    else
	info "Building outside docker..."
    fi

    case $1 in
	b|build)
	    shift
	    cmd_build $@
	    ;;
	c|clean)
	    cmd_clean
	    ;;
	*)
	    fatal Unknown command \'$1\'. Use -h or --help.
	    ;;
    esac
}

main $@
