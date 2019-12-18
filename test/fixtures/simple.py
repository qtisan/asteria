import pytest
import moment


@pytest.fixture(scope='module')
def long_str():
    return 'Fixtures requiring network access depend on connectivity and are usually time-expensive to create. Extending the previous example, we can add a scope="module" parameter to the @pytest.fixture invocation to cause the decorated smtp_connection fixture function to only be invoked once per test module (the default is to invoke once per test function). Multiple test functions in a test module will thus each receive the same smtp_connection fixture instance, thus saving time. Possible values for scope are: function, class, module, package or session.'


@pytest.fixture()
def now():
    return moment.now()