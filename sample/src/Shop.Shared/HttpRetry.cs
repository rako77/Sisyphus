namespace Shop.Shared;

public static class HttpRetry
{
    public static async Task<T> RunAsync<T>(Func<Task<T>> call, ILogger log, int maxAttempts = 4)
    {
        var backoff = TimeSpan.FromMilliseconds(250);
        for (var attempt = 1; ; attempt++)
        {
            try
            {
                return await call().ConfigureAwait(false);
            }
            catch (HttpRequestException ex) when (attempt < maxAttempts)
            {
                log.Warn($"Attempt {attempt} failed: {ex.Message}");
                await Task.Delay(backoff).ConfigureAwait(false);
                backoff = TimeSpan.FromMilliseconds(Math.Min(backoff.TotalMilliseconds * 2, 5000));
            }
        }
    }
}
