package postgresql

import (
	"context"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/stretchr/testify/assert"
)

func TestApplyEngineOptionsConfig(t *testing.T) {

	testCases := []struct {
		name       string
		opts       []Option
		wantErr    bool
		wantIpType IpType
	}{
		{
			name: "valid config with connection pool",
			opts: []Option{
				WithPool(&pgxpool.Pool{}),
				WithDatabase("testdb"),
			},
			wantErr:    false,
			wantIpType: PUBLIC,
		},
		{
			name: "valid config with instance details",
			opts: []Option{
				WithCloudSQLInstance("testproject", "testregion", "testinstance"),
				WithDatabase("testdb"),
			},
			wantErr:    false,
			wantIpType: PUBLIC,
		},
		{
			name: "missing database",
			opts: []Option{
				WithCloudSQLInstance("testproject", "testregion", "testinstance"),
			},
			wantErr:    true,
			wantIpType: PUBLIC,
		},
		{
			name: "missing all connection details",
			opts: []Option{
				WithDatabase("testdb"),
			},
			wantErr:    true,
			wantIpType: PUBLIC,
		},
		{
			name: "ip type private",
			opts: []Option{
				WithCloudSQLInstance("testproject", "testregion", "testinstance"),
				WithDatabase("testdb"),
				WithIPType(PRIVATE),
			},
			wantErr:    false,
			wantIpType: PRIVATE,
		},
		{
			name: "custom EmailRetriever",
			opts: []Option{
				WithCloudSQLInstance("testproject", "testregion", "testinstance"),
				WithDatabase("testdb"),
			},
			wantErr:    false,
			wantIpType: PUBLIC,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := applyEngineOptions(tc.opts)
			if tc.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.wantIpType, cfg.ipType)
			}
		})
	}
}

func TestGetUser(t *testing.T) {
	testCases := []struct {
		name        string
		cfg         engineConfig
		wantUser    string
		wantIAMAuth bool
		wantErr     bool
	}{
		{
			name: "user and password provided",
			cfg: engineConfig{
				user:     "testuser",
				password: "testpassword",
			},
			wantUser:    "testuser",
			wantIAMAuth: false,
			wantErr:     false,
		},
		{
			name: "iam account email provided",
			cfg: engineConfig{
				iamAccountEmail: "iam@example.com",
			},
			wantUser:    "iam@example.com",
			wantIAMAuth: true,
			wantErr:     false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			user, iamAuth, err := getUser(ctx, tc.cfg)
			if tc.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.wantUser, user)
				assert.Equal(t, tc.wantIAMAuth, iamAuth)
			}
		})
	}
}
