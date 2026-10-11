static bool read_token_file(const std::string & file_name, const llama_vocab * vocab,
                            std::vector<llama_token> & tokens) {
    std::ifstream input(file_name, std::ios::binary);
    if (!input) {
        LOG_ERR("%s: failed to open token file %s\n", __func__, file_name.c_str());
        return false;
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0 || size % sizeof(int32_t) != 0) {
        LOG_ERR("%s: token file %s must contain whole little-endian int32 values\n", __func__, file_name.c_str());
        return false;
    }
    input.seekg(0, std::ios::beg);
    std::vector<unsigned char> bytes(static_cast<size_t>(size));
    input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) {
        LOG_ERR("%s: failed to read token file %s\n", __func__, file_name.c_str());
        return false;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    tokens.resize(bytes.size() / sizeof(int32_t));
    for (size_t i = 0; i < tokens.size(); ++i) {
        const uint32_t id = static_cast<uint32_t>(bytes[4*i]) |
                            (static_cast<uint32_t>(bytes[4*i + 1]) << 8) |
                            (static_cast<uint32_t>(bytes[4*i + 2]) << 16) |
                            (static_cast<uint32_t>(bytes[4*i + 3]) << 24);
        if (id >= static_cast<uint32_t>(n_vocab)) {
            LOG_ERR("%s: token id %u at offset %zu is outside the model vocabulary (0..%d)\n",
                    __func__, id, 4*i, n_vocab - 1);
            return false;
        }
        tokens[i] = static_cast<llama_token>(id);
    }
    LOG_INF("%s: loaded %zu token ids from %s\n", __func__, tokens.size(), file_name.c_str());
    return true;
}

static bool extract_token_file_argument(int & argc, char ** argv) {
    int out = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        std::string value;
        if (argument == "--token-file") {
            if (++i >= argc) {
                LOG_ERR("%s: --token-file requires a file name\n", __func__);
                return false;
            }
            value = argv[i];
        } else if (argument.rfind("--token-file=", 0) == 0) {
            value = argument.substr(argument.find('=') + 1);
        } else {
            argv[out++] = argv[i];
            continue;
        }
        if (value.empty() || !g_token_file.empty()) {
            LOG_ERR("%s: --token-file requires exactly one nonempty file name\n", __func__);
            return false;
        }
        g_token_file = value;
    }
    argv[out] = nullptr;
    argc = out;
    return true;
}
